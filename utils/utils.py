
from torch.nn import functional
import json
import argparse
import torch
import difflib
import logging

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer
import csv

def collate_wiki(instances,tokenizer,return_attention_masks= True, pad_to_max_length = False, device='cuda'):

    token_ids = [tokenizer.encode(_x[1], max_length=200) for _x in instances]

    if pad_to_max_length:
        batch_max_len = 512
    else:
        batch_max_len = max([len(_s) for _s in token_ids])
    padded_ids_tensor = torch.tensor(
        [_s + [tokenizer.pad_token_id] * (batch_max_len - len(_s)) for _s in
         token_ids])

    labels = torch.tensor([int(_x[0]) for _x in instances], dtype=torch.long)

    output_tensors = [padded_ids_tensor]
    if return_attention_masks:
        output_tensors.append(padded_ids_tensor > 0)
    output_tensors.append(labels)

    return list(_t.to(device) for _t in output_tensors)


# class logger(self,log_path,logging_name):
def logger_config(log_path,logging_name):

    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def calculate_aul(model, token_ids, log_softmax, attention):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AULA or AUL).
    '''
    output = model(token_ids)
    logits = output.logits.squeeze(0)
    log_probs = log_softmax(logits)
    token_ids = token_ids.view(-1, 1).detach()
    token_log_probs = log_probs.gather(1, token_ids)[1:-1]
    if attention:
        attentions = torch.mean(torch.cat(output.attentions, 0), 0)
        averaged_attentions = torch.mean(attentions, 0)
        averaged_token_attentions = torch.mean(averaged_attentions, 0)
        token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]
    sentence_log_prob = torch.mean(token_log_probs)
    score = sentence_log_prob.item()

    ranks = get_rank_for_gold_token(log_probs, token_ids)

    return score, ranks

