"""Script to serialize the saliency with gradient approaches and occlusion."""
import argparse
import json
import os
import random
from argparse import Namespace
from collections import defaultdict
from functools import partial

import numpy as np
import torch
# from captum.attr import ShapleyValueSampling
from pypapi import events, papi_high as high
from captum.attr import DeepLift, GuidedBackprop, InputXGradient, Occlusion, \
    Saliency, configure_interpretable_embedding_layer, \
    remove_interpretable_embedding_layer
# from pypapi import events, papi_high as high
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizer
from transformers import BertTokenizer, BertModel, BertConfig

from utils import *
from model import *
from Dataset import *
import os


def summarize_attributions(attributions, type='mean', model=None, tokens=None):
    if type == 'none':
        return attributions
    elif type == 'dot':
        embeddings = get_model_embedding_emb(model)(tokens)
        attributions = torch.einsum('bwd, bwd->bw', attributions, embeddings)
    elif type == 'mean':
        attributions = attributions.mean(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
    elif type == 'l2':
        attributions = attributions.norm(p=1, dim=-1).squeeze(0)
    return attributions


class BertModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, attention_mask):
        return self.model(input, attention_mask=attention_mask)[0]

def get_model_embedding_emb(model):
    if args.model == 'trans':
        return model.bert.embeddings.embedding.word_embeddings
    else:
        return model.embedding.embedding


def generate_saliency(model_path, saliency_path, saliency, aggregation):
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    # model_args = Namespace(**checkpoint['args'])
    # if args.model == 'lstm':
    #     model = LSTM_MODEL(tokenizer, model_args,
    #                        n_labels=checkpoint['args']['labels']).to(device)
    #     model.load_state_dict(checkpoint['model'])
    # elif args.model == 'trans':
    #     transformer_config = BertConfig.from_pretrained('bert-base-uncased',
    #                                                     num_labels=model_args.labels)
    #     model_cp = BertForSequenceClassification.from_pretrained(
    #         'bert-base-uncased', config=transformer_config).to(
    #         device)
    #     checkpoint = torch.load(model_path,
    #                             map_location=lambda storage, loc: storage)
    #     model_cp.load_state_dict(checkpoint['model'])
    model_cp = checkpoint.to(device)
    model = BertModelWrapper(model_cp)
    # else:
    #     model = CNN_MODEL(tokenizer, model_args,
    #                       n_labels=checkpoint['args']['labels']).to(device)
    #     model.load_state_dict(checkpoint['model'])

    model.eval()

    pad_to_max = False
    if saliency == 'deeplift':
        ablator = DeepLift(model)
    elif saliency == 'guided':
        ablator = GuidedBackprop(model)
    elif saliency == 'sal':
        ablator = Saliency(model)
    elif saliency == 'inputx':
        ablator = InputXGradient(model)
    elif saliency == 'occlusion':
        ablator = Occlusion(model)

    coll_call = collate_wiki

    # return_attention_masks = args.model == 'trans'

    collate_fn = partial(coll_call, tokenizer=tokenizer, device=device,
                         return_attention_masks=True,
                         pad_to_max_length=pad_to_max)
    test = Wiki_v2_Dataset(args.dataset_dir, type='test')
    # get_dataset(path=args.dataset_dir, mode=args.split,
                    #    dataset=args.dataset)
    batch_size = args.batch_size
    test_dl = DataLoader(batch_size=batch_size, dataset=test, shuffle=False,
                         collate_fn=collate_fn)

    # PREDICTIONS
    predictions_path = model_path + '.predictions_occ'
    if not os.path.exists(predictions_path):
        predictions = defaultdict(lambda: [])
        for batch in tqdm(test_dl, desc='Running test prediction... '):
            # print("*************111",batch)
            logits = model(batch[0], attention_mask=batch[1])
            logits = logits.detach().cpu().numpy().tolist()
            predicted = np.argmax(np.array(logits), axis=-1)
            predictions['class'] += predicted.tolist()
            predictions['logits'] += logits

        with open(predictions_path, 'w') as out:
            json.dump(predictions, out)

    # COMPUTE SALIENCY
    if saliency != 'occlusion':
        embedding_layer_name = 'model.bert.embeddings' if args.model == \
                                                          'trans' else \
            'embedding'
        interpretable_embedding = configure_interpretable_embedding_layer(model,
                                                                          embedding_layer_name)

    class_attr_list = defaultdict(lambda: [])
    token_ids = []
    saliency_flops = []

    for batch in tqdm(test_dl, desc='Running Saliency Generation...'):
        # if args.model == 'cnn':
        #     additional = None
        # elif args.model == 'trans':
        #     # print(batch[2])
        additional = batch[1]
        # else:
        #     additional = batch[-1]

        token_ids += batch[0].detach().cpu().numpy().tolist()
        if saliency != 'occlusion':
            # print(batch[0])
            input_embeddings = interpretable_embedding.indices_to_embeddings(batch[0])

        # if not args.no_time:
        #     high.start_counters([events.PAPI_FP_OPS])
        for cls_ in range(2):
            if saliency == 'occlusion':
                attributions = ablator.attribute(batch[0],
                                                 sliding_window_shapes=(
                                                 args.sw,), target=cls_,
                                                 additional_forward_args=additional)
            else:
                attributions = ablator.attribute(input_embeddings, target=cls_,
                                                 additional_forward_args=additional)

            attributions = summarize_attributions(attributions,
                                                  type=aggregation, model=model,
                                                  tokens=batch[0]).detach().cpu().numpy().tolist()
            class_attr_list[cls_] += [[_li for _li in _l] for _l in
                                      attributions]

        # if not args.no_time:
        #     print(args.no_time)
        #     saliency_flops.append(sum(high.stop_counters()) / batch[0].shape[0])

    if saliency != 'occlusion':
        remove_interpretable_embedding_layer(model, interpretable_embedding)

    # SERIALIZE
    print('Serializing...', flush=True)
    with open(saliency_path, 'w') as out:
        for instance_i, _ in enumerate(test):
            saliencies = []
            for token_i, token_id in enumerate(token_ids[instance_i]):
                token_sal = {'token': tokenizer.ids_to_tokens[token_id]}
                for cls_ in range(2):
                    token_sal[int(cls_)] = class_attr_list[cls_][instance_i][token_i]
                saliencies.append(token_sal)

            out.write(json.dumps({'tokens': saliencies}) + '\n')
            out.flush()

    return saliency_flops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Which dataset", default='snli',
                        type=str, choices=['snli', 'imdb', 'tweet'])
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/',
                        type=str)
    parser.add_argument("--split", help="Which split of the dataset",
                        default='test', type=str,
                        choices=['train', 'test'])
    parser.add_argument("--no_time",
                        help="Whether to output the time for generation in "
                             "flop",
                        action='store_true',        
                        default=False)
    parser.add_argument("--model", help="Which model", default='cnn',
                        choices=['cnn', 'lstm', 'trans'], type=str)
    parser.add_argument("--models_dir",
                        help="Path where the models can be found, "
                             "with a common prefix, without _1",
                        default='snli_bert', type=str)
    parser.add_argument("--gpu", help="Flag for running on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--output_dir",
                        help="Path where the saliencies will be serialized",
                        default='saliency_snli',
                        type=str)
    parser.add_argument("--sw", help="Sliding window", type=int, default=1)
    parser.add_argument("--saliency", help="Saliency type", nargs='+')
    parser.add_argument("--batch_size",
                        help="Batch size for explanation generation", type=int,
                        default=None)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--p_dropout', type=float, default=0.3)
    parser.add_argument('--out_dim', type=int, default=2 )
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.99)
    parser.add_argument('--num_layer', type=int, default=12)
    # parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=100)
    # parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--out_attention', type=str,default='')
    # parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--fc_dim', type=int, default=300)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    print(args, flush=True)

    for saliency in args.saliency:
        print('Running Saliency ', saliency, flush=True)

        if saliency in ['guided', 'sal', 'inputx', 'deeplift']:
            aggregations = ['mean', 'l2']  #
        else:  # occlusion
            aggregations = ['none']

        for aggregation in aggregations:
            flops = []
            print('Running aggregation ', aggregation, flush=True)

            models_dir = args.models_dir
            base_model_name = models_dir.split('/')[-1]
            # n = 6, change here
            # for model in range(1, 2):
            curr_flops = generate_saliency(
                models_dir,
                os.path.join(args.output_dir,
                                f'{base_model_name}_{saliency}_{aggregation}'),
                saliency,
                aggregation)
            print(curr_flops)
            flops.append(np.average(curr_flops))

            print('FLOPS', np.average(flops), np.std(flops), flush=True)