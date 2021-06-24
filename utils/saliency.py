"""Script to serialize the saliency scored produced by the Shapley Values
Sampling"""
import argparse
import json
import os
import random
from argparse import Namespace
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from captum.attr import ShapleyValueSampling
from pypapi import events, papi_high as high
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizer
from transformers import BertTokenizer, BertModel, BertConfig

from utils import *
from model import *
from Dataset import *
import os
# from models.data_loader import IMDBDataset, NLIDataset, TwitterDataset, \
#     get_collate_fn
# from models.model_builder import CNN_MODEL, LSTM_MODEL

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        return self.model(input.long())


class BertModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        return self.model(input.long(), attention_mask=input > 0)[0]


# def get_dataset(path, mode='test'):
#     if args.dataset == 'snli':
#         ds = NLIDataset(path, type=mode, salient_features=True)
#     elif args.dataset == 'imdb':
#         ds = IMDBDataset(path, type=mode)
#     elif args.dataset == 'tweet':
#         ds = TwitterDataset(path, type=mode)
#     return ds


def generate_saliency(model_path, saliency_path):
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    # print(checkpoint['args'])
    # model_args = Namespace(**checkpoint['args'])
    # model_args.batch_size = args.batch_size if args.batch_size != None else \
    #     model_args.batch_size

    # if args.model == 'transformer':
    # transformer_config = BertConfig.from_pretrained('bert-base-uncased',
    #                                                 num_labels=model_args.labels)
    # modelb = BertForSequenceClassification.from_pretrained(
    #     'bert-base-uncased', config=transformer_config).to(
    #     device)
    
    # model = BERTClassifier(encoder,args)
    # configuration = BertConfig.from_pretrained("bert-base-multilingual-cased")
    # configuration.num_hidden_layers = 1
    # configuration.output_attentions=True
    # encoder = BertModel.from_pretrained("bert-base-multilingual-cased", config = configuration)    
    # # modelb = BERTClassifier()
    # modelb = BERTClassifier(encoder,args).to(device)
    # modelb.load_state_dict(checkpoint['model'])
    modelb = checkpoint.to(device)
    model = BertModelWrapper(modelb)
    # else:
    #     # model_args.batch_size = 1000
    #     model = CNN_MODEL(tokenizer, model_args,
    #                       n_labels=checkpoint['args']['labels']).to(device)
    #     model.load_state_dict(checkpoint['model'])
    #     model.train()
    #     model = ModelWrapper(model)

    ablator = ShapleyValueSampling(model)

    coll_call = collate_wiki
    # get_collate_fn(dataset=args.dataset, model=args.model)

    collate_fn = partial(coll_call, tokenizer=tokenizer, device=device,
                         return_attention_masks=False,
                         pad_to_max_length=False)

    test = Wiki_v2_Dataset(args.dataset_dir, type='test')
    # get_dataset(args.dataset_dir, mode=args.split)
    test_dl = DataLoader(batch_size=args.batch_size, dataset=test,
                         shuffle=False, collate_fn=collate_fn)
    # print(test_dl)

    # PREDICTIONS
    predictions_path = model_path + '.predictions'
    print(predictions_path)
    if not os.path.exists(predictions_path):
        predictions = defaultdict(lambda: [])
        for batch in tqdm(test_dl, desc='Running test prediction... '):
            # print(batch)
            logits = model(batch[0])
            logits = logits.detach().cpu().numpy().tolist()
            predicted = np.argmax(np.array(logits), axis=-1)
            predictions['class'] += predicted.tolist()
            predictions['logits'] += logits

        with open(predictions_path, 'w') as out:
            json.dump(predictions, out)

    # COMPUTE SALIENCY

    saliency_flops = []

    with open(saliency_path, 'w') as out_mean:
        for batch in tqdm(test_dl, desc='Running Saliency Generation...'):
            class_attr_list = defaultdict(lambda: [])

            # if args.model == 'rnn':
            #     additional = batch[-1]
            # else:
            additional = None

            if not args.no_time:
                high.start_counters([events.PAPI_FP_OPS])
            token_ids = batch[0].detach().cpu().numpy().tolist()

            for cls_ in range(args.labels):
                attributions = ablator.attribute(batch[0].float(), target=cls_,
                                                 additional_forward_args=additional)
                attributions = attributions.detach().cpu().numpy().tolist()
                class_attr_list[cls_] += attributions

            if not args.no_time:
                x = sum(high.stop_counters())
                saliency_flops.append(x / batch[0].shape[0])

            for i in range(len(batch[0])):
                saliencies = []
                for token_i, token_id in enumerate(token_ids[i]):
                    if token_id == tokenizer.pad_token_id:
                        continue
                    token_sal = {'token': tokenizer.ids_to_tokens[token_id]}
                    for cls_ in range(args.labels):
                        token_sal[int(cls_)] = class_attr_list[cls_][i][token_i]
                    saliencies.append(token_sal)

                out_mean.write(json.dumps({'tokens': saliencies})+ '\n')
                out_mean.flush()

    return saliency_flops





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/',
                        type=str)
    # parser.add_argument("--dataset", help="Which dataset", default='snli',
    #                     type=str, choices=['snli', 'imdb', 'tweet'])
    parser.add_argument("--split", help="Which split of the dataset",
                        default='test', type=str,
                        choices=['train', 'test'])
    parser.add_argument("--no_time",
                        help="Whether to output the time for generation in "
                             "flop",
                        action='store_true',
                        default=False)
    # parser.add_argument("--model", help="Which model", default='cnn',
    #                     choices=['cnn', 'lstm', 'transformer'], type=str)
    parser.add_argument("--gpu", help="Flag for running on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="Num of labels", type=int, default=3)
    parser.add_argument("--model_path",
                        help="Path to the model", default=None,
                        type=str)
    parser.add_argument("--output_dir",
                        help="Path where the saliencies will be serialized",
                        default='saliency_snli_lime',
                        type=str)
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

    model_path = args.model_path
    model_name = model_path.split('/')[-1]

    all_flops = generate_saliency(model_path, os.path.join(args.output_dir,
                                                           f'{model_name}_shap'))

    print('FLOPS', np.average(all_flops), np.std(all_flops), flush=True)
