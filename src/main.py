from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW

from pathlib import Path
import csv
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler
from torch.nn.parallel import DistributedDataParallel
from model import BERTClassifier
from Dataset import WikiDataset
import os
import argparse
from utils import mask_cross_entropy,logger_config

import torch
import sys
import torch.nn as nn
import numpy as np
import random
import glob
import logging
import pickle
import time

# torch.distributed.init_process_group(backend='nccl')

def read_wiki_split(filepath):
    # split_dir = Path(split_dir)
    texts = []
    labels = []

    with open(filepath, 'r') as files:
        reader = csv.reader(files, delimiter='\n')
        for line in tqdm(reader):
            texts.append(line[0][:-2])
            labels.append(int(line[0][-1]))

    return texts, labels

def save_attention(attention, output_file):
    with open(output_file, 'w') as f:
        torch.save(attention, f)

def main(args):
    # print(os.path.join(args.input_path,'dev.txt'))
    train_texts, train_labels = read_wiki_split(args.input_path_train)
    test_texts, test_labels = read_wiki_split(args.input_path_valid)
    val_texts, val_labels = read_wiki_split(args.input_path_test)
    # print('****35', val_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    train_encodings = tokenizer(train_texts, max_length=100, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, max_length=100,padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, max_length=100,padding=True)
    # print('****test', test_encodings)

    train_dataset = WikiDataset(train_encodings, train_labels)
    val_dataset = WikiDataset(val_encodings, val_labels)
    test_dataset = WikiDataset(test_encodings, test_labels)
    # train the model
    # t = time.strftime("%Y-%m-%d %H", time.localtime())
    # t='lan0'
    t = args.input_path_test.split('/')[-1]
    logger = logger_config(log_path='./logs/log_{}.txt'.format(t), logging_name='records')
    trainer(args, train_dataset, val_dataset, test_dataset,logger)


def trainer(args, train_dataset, val_dataset, test_dataset,logger=None):
    
    for param in encoder.base_model.parameters():
        param.requires_grad = True

    if not args.if_specific:
        # print('here')
        logger.info('all layers before {} are used'.format(args.num_layer))
        configuration = BertConfig.from_pretrained("bert-base-multilingual-cased")
        configuration.num_hidden_layers = args.num_layer
        configuration.output_attentions=True
        encoder = BertModel.from_pretrained("bert-base-multilingual-cased", config = configuration)
    else:
        logger.info('only {}th layer is used'.format(args.k))
        configuration = BertConfig.from_pretrained("bert-base-multilingual-cased")
        configuration.num_hidden_layers = 1
        configuration.output_attentions=True
        bert_base_cased = BertModel.from_pretrained("bert-base-multilingual-cased",config = configuration)

        encoder = BertModel(config = configuration)
        encoder.base_model.encoder.layer[0] = bert_base_cased.base_model.encoder.layer[args.k]

        


    print('config', configuration)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    # optim = AdamW(model.parameters(), lr=5e-5)

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # random.seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    
    model = BERTClassifier(encoder,args)
    model.to(device)
    optim = AdamW(model.parameters(),
                            lr=0.0005,
                            weight_decay=args.weight_decay)
    model.train()

    best_dev_acc = -1

    

    for epoch in range(args.num_epoch):
        all_loss = []
        n_train_correct = 0
        n_total = 0
        # train_loader.init_epoch()
        for batch_idx, batch in enumerate(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits, attention, last_hidden = model(input_ids=input_ids, attention_mask=attention_mask)
            # loss = outputs[0]
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits,labels)
            all_loss.append(loss.item())
            loss.backward()
            optim.step()
            n_train_correct +=(torch.max(logits, 1)[1].view(labels.size()).data==labels.data).sum()
            n_total += args.batch_size
        # print('correct', n_train_correct.item(), n_total)

        train_acc = 100 * n_train_correct.cpu().numpy()/n_total

        logger.info('Training: Progress:{epoch_idx}/{num_epoch}, accuracy:{acc}, loss:{loss}'.format(epoch_idx = epoch, num_epoch=args.num_epoch, acc=train_acc, loss= np.mean(all_loss)))

        # validation process here
        model.eval()
        n_val_correct = 0
        n_total_val = 0 
        val_loss = 0
        val_losses = []
        tmp = []
        for val_batch_idx, val_batch in enumerate(val_loader):
            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            labels = val_batch['labels'].to(device)
            logits, attention, last_hidden = model(input_ids=input_ids, attention_mask=attention_mask)
            criterion = nn.CrossEntropyLoss()
            val_loss = criterion(logits, labels)
            val_losses.append(val_loss.item())
            n_val_correct += (torch.max(logits, 1)[1].view(labels.size()).data==labels.data).sum()
            n_total_val += args.batch_size
            tmp.append(logits)


        val_acc = 100. * n_val_correct.cpu().numpy() / n_total_val        
        if val_acc > best_dev_acc:
            best_dev_acc = val_acc
            print(best_dev_acc)
            if args.model_path:
                t = args.input_path_test.split('/')[-1]
                snapshot_prefix = os.path.join(args.model_path, 'best')
                dev_snapshot_path = snapshot_prefix + \
                                    '_devacc_{}_epoch_{}_{}_{}.pt'.format(val_acc, 1 + epoch, configuration.num_hidden_layers, t)

                torch.save(model, dev_snapshot_path)

        logger.info('Validation: Progress:{epoch}/{num_epoch}, accuracy:{acc}'.format(epoch=epoch, num_epoch=args.num_epoch, acc=val_acc))

    # for test dataset only
    test_model = torch.load(dev_snapshot_path)
    # Switch model to evaluation mode
    test_model.eval()

    n_test_correct = 0
    n_total_test = 0
    test_loss = 0
    test_losses = []

    for test_batch_idx, test_batch in enumerate(test_loader):
            input_ids = test_batch['input_ids'].to(device)
            attention_mask = test_batch['attention_mask'].to(device)
            labels = test_batch['labels'].to(device)
            logits, attention, last_hidden = model(input_ids=input_ids, attention_mask=attention_mask)
            criterion = nn.CrossEntropyLoss()
            test_loss = criterion(logits, labels)
            test_losses.append(test_loss.item())
            n_test_correct += (torch.max(logits, 1)[1].view(labels.size()).data==labels.data).sum()
            n_total_test += args.batch_size

    test_acc = 100. * n_test_correct.cpu().numpy() / n_total_test
    logger.info('Test: Progress:{epoch}/{num_epoch}, accuracy:{acc}'.format(epoch=epoch, num_epoch=args.num_epoch, acc=val_acc))

        
if __name__ == "__main__":
    argparse = argparse.ArgumentParser(sys.argv[0])
    argparse.add_argument('--input_path_train',  type=str, default=None)
    argparse.add_argument('--input_path_valid',  type=str, default=None)
    argparse.add_argument('--input_path_test',  type=str, default=None)
    argparse.add_argument('--hidden_dim', type=int, default=768)
    argparse.add_argument('--p_dropout', type=float, default=0.3)
    argparse.add_argument('--out_dim', type=int, default=2 )
    argparse.add_argument('--num_epoch', type=int, default=100)
    argparse.add_argument('--weight_decay', type=float, default=0.99)
    argparse.add_argument('--num_layer', type=int, default=12)
    argparse.add_argument('--batch_size', type=int, default=64)
    argparse.add_argument('--max_length', type=int, default=100)
    argparse.add_argument('--model_path', type=str, default=None)
    argparse.add_argument('--out_attention', type=str,default='')
    argparse.add_argument('--seed', type=int, default=20)
    argparse.add_argument('--fc_dim', type=int, default=300)
    argparse.add_argument('--save_attention', action='store_true')
    argparse.add_argument('--save_salient', type=str, default=None)
    argparse.add_argument('--if_specific', action='store_true')
    argparse.add_argument('--k',type=int,default=0)
    args = argparse.parse_args()
    print(args)
    main(args)

