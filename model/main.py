import time
import sys
import argparse
import random
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from utils.metric import get_ner_fmeasure
# from model.seqlabel import SeqLabel
# from model.sentclassifier import SentClassifier
from utils.data import Data
from model.corefmodel import CoRef
from utils.write_result import write
import pandas as pd
import json
from tqdm import tqdm
# sys.path.append('../JointParser')
from JointParser.pair_infer import joint_infer
import os
import datetime
import argparse



def singlefy_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):

    batch_size = len(input_batch_list)
    words = [input_batch_list[0][0]]
    token_labels = [input_batch_list[0][4]]
    chars = [input_batch_list[0][1]]
    combines = [input_batch_list[0][2]]
    combine_labels = [input_batch_list[0][3]]
    word_features = [input_batch_list[0][5]]

    word_seq_length = torch.LongTensor([len(words[0])])

    word_seq_tensor = torch.zeros((1, word_seq_length), requires_grad = if_train).long()
    for idx, (seq, seqlen) in enumerate(zip(words, word_seq_length)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    pad_chars = [chars[idx] + [[0]] * (word_seq_length[0]-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, word_seq_length[0], max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[0].view(batch_size*word_seq_length[0],-1)
    char_seq_lengths = char_seq_lengths[0].view(batch_size*word_seq_length[0],)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    word_seq_recover = [0]

    token_labels_tensor = torch.tensor(token_labels)
    word_features_tensor = torch.tensor(word_features).float()
    # print(word_features_tensor.size())
    # if gpu:
    #     word_seq_tensor = word_seq_tensor.cuda()
    #     for idx in range(feature_num):
    #         feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
    #     word_seq_lengths = word_seq_lengths.cuda()
    #     word_seq_recover = word_seq_recover.cuda()
    #     label_seq_tensor = label_seq_tensor.cuda()
    #     char_seq_tensor = char_seq_tensor.cuda()
    #     char_seq_recover = char_seq_recover.cuda()
    #     mask = mask.cuda()
    return word_seq_tensor, word_seq_length, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, combines, combine_labels, token_labels_tensor, word_features_tensor

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train(data, model):
    print("Training model...")

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    for idx in range(data.HP_iteration):
        model.train()

        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        batch_size = 1
        batch_id = 0
        train_num = len(data.train_Ids)
        # total_batch = train_num//batch_size+1
        total_batch = train_num
        print("Batch num:", total_batch)
        total_loss = 0
        full_loss = 0
        
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_combine, batch_combine_labels, token_labels, word_features = singlefy_sequence_labeling_with_label(instance, data.HP_gpu, True)
            for combine, combine_label in zip(batch_combine[0], batch_combine_labels[0]):
                loss = model.compute_single_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, combine, combine_label, token_labels, word_features)
                loss.backward()
                full_loss += loss.item()
                optimizer.step()
                model.zero_grad()

        print('Loss:', full_loss)

def test(data, model):

    model.eval()
    test_num = len(data.test_Ids)
    # total_batch = train_num//batch_size+1
    batch_size = 1
    total_batch = test_num

    outputs = []

    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > test_num:
            end = test_num
        instance = data.test_Ids[start:end]
        if not instance:
            continue

        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_combine, batch_combine_labels, token_labels, word_features = singlefy_sequence_labeling_with_label(instance, data.HP_gpu, False)
        predicts, predict_token_label = model.predict(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_combine, batch_combine_labels, word_features)

        sentence = data.test_texts[start:end][0][0]

        outputs.append(write(sentence, predicts, predict_token_label))

    #store as csv

    # df_outputs = pd.DataFrame(outputs)
    # df_outputs.to_csv(store_path, index=False)
    return outputs



def data_initialization(data, word_emb_dir):
    # data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir, word_emb_dir)
    # data.build_alphabet(data.dev_dir)
    # data.build_alphabet(data.test_dir)
    data.fix_alphabet()

def train_base_model():

    word_emb_dir = 'w2v.emb'
    train_dir = '../data/train/annotated_log.v1.train'
    data = Data(train_dir, word_emb_dir)
    data_initialization(data, word_emb_dir)
    data.build_pretrain_emb()
    data.generate_instance('train')
    data.HP_iteration = 50


    model = CoRef(data)

    train(data, model)
    torch.save(model, 'basemodel.pkl')


def finetune_model(train_dir, word_emb_dir, test_dir, model, finetuneset=None):

    data = Data(train_dir, word_emb_dir)
    data.generate_instance('finetune')
    data.HP_iteration = 30
    train(data, model)
    torch.save(model, '{}_finetune_model_820.pkl'.format(finetuneset))

    data.test_dir = test_dir
    data.generate_instance('test')
    outputs = test(data, model)
    print(outputs)

    # write results
    write_dir = '../data/Prediction'

    #formulate data to csv
    for idx, output in enumerate(outputs):
        # print(len(output['log']))
        outputs[idx]['log'] = output['log'][1:]
        # print(len(outputs[idx]['log']))
        outputs[idx]['concept'] = [i - 1 for i in output['concept']]
        outputs[idx]['instance'] = [i - 1 for i in output['instance']]
        tmp_pairs = []
        for pair in output['pairs']:
            tmp_pairs.append([pair[0] - 1, pair[1] - 1])
        outputs[idx]['pairs'] = tmp_pairs
        # print(outputs)

        # print(outputs)
        # entry: log, pairs, concept, instance

    logs = [item['log'] for item in outputs]
    pairs = [item['pairs'] for item in outputs]
    concepts = [item['concept'] for item in outputs]
    instances = [item['instance'] for item in outputs]

    structured_result = []

    real_pair, left_concept, left_instance, conceptualized, params = joint_infer(logs, pairs, concepts, instances)
    print('Inference END!')
    line_id = 1
    for log, pair, concept, instance, ctemplate, param in zip(logs, real_pair, concepts, instances, conceptualized, params):
        tmp_structure_log = {}
        tmp_structure_log['LineID'] = line_id
        tmp_structure_log['log'] = log
        tmp_structure_log['pair'] = pair
        tmp_structure_log['concept'] = [log[i] for i in concept]
        tmp_structure_log['instance'] = [log[i] for i in instance]
        tmp_structure_log['conceptualized'] = ctemplate
        tmp_structure_log['parameter'] = param
        structured_result.append(tmp_structure_log)
        line_id += 1

    df_strctured_result = pd.DataFrame(structured_result)
    df_strctured_result.to_csv(os.path.join(write_dir, '{}.csv'.format(finetuneset)), index=False)
    print('Save to ', os.path.join(write_dir, '{}.csv'.format(finetuneset)))



def test_model(model, word_emb_dir, test_dir, finetuneset):

    data = Data(train_dir=None, word_emb_dir=None)
    data.test_dir = test_dir

    data.generate_instance('test')
    start_time = datetime.datetime.now()
    outputs = test(data, model)
    end_time = datetime.datetime.now()
    print(end_time-start_time).seconds()

    write_dir = '../data/prediction'

    #formulate data to csv
    for idx, output in enumerate(outputs):
        # print(len(output['log']))
        outputs[idx]['log'] = output['log'][1:]
        # print(len(outputs[idx]['log']))
        outputs[idx]['concept'] = [i - 1 for i in output['concept']]
        outputs[idx]['instance'] = [i - 1 for i in output['instance']]
        tmp_pairs = []
        for pair in output['pairs']:
            tmp_pairs.append([pair[0] - 1, pair[1] - 1])
        outputs[idx]['pairs'] = tmp_pairs
        # print(outputs)

        # print(outputs)
        # entry: log, pairs, concept, instance

    logs = [item['log'] for item in outputs]
    pairs = [item['pairs'] for item in outputs]
    concepts = [item['concept'] for item in outputs]
    instances = [item['instance'] for item in outputs]

    structured_result = []

    real_pair, left_concept, left_instance, conceptualized, params = joint_infer(logs, pairs, concepts, instances)
    print('Inference END!')
    line_id = 1
    for log, pair, concept, instance, ctemplate, param in zip(logs, real_pair, concepts, instances, conceptualized, params):
        tmp_structure_log = {}
        tmp_structure_log['LineID'] = line_id
        tmp_structure_log['log'] = log
        tmp_structure_log['pair'] = pair
        tmp_structure_log['concept'] = [log[i] for i in concept]
        tmp_structure_log['instance'] = [log[i] for i in instance]
        tmp_structure_log['conceptualized'] = ctemplate
        tmp_structure_log['parameter'] = param
        structured_result.append(tmp_structure_log)
        line_id += 1

    df_strctured_result = pd.DataFrame(structured_result)
    df_strctured_result.to_csv(os.path.join(write_dir, '{}.csv'.format(finetuneset)), index=False)
    print('Save to ', os.path.join(write_dir, '{}.csv'.format(finetuneset)))


    return outputs



parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="test", type=str, choices=["train", "finetune", "test"])
params = vars(parser.parse_args())

if params["mode"] == "train":
    train_base_model()


if params["mode"] == "finetune":
    for finetune_set in ['Andriod', 'BGL', 'Hadoop', 'HDFS', 'Linux', 'OpenStack','Spark', 'Zookeeper']:
        word_emb_dir = 'w2v.emb'
        test_dir = '../data/test/{}.log'.format(finetune_set)
        model = torch.load('basemodel.pkl')
        finetune_model(train_dir, word_emb_dir, test_dir, model, finetuneset=finetune_set)
        outputs = test_model(model, test_dir)

if params["mode"] == "test":
    for finetune_set in ['Andriod', 'BGL', 'Hadoop', 'HDFS', 'Linux', 'OpenStack','Spark', 'Zookeeper']:
        word_emb_dir = 'w2v.emb'
        test_dir = '../data/test/{}.log'.format(finetune_set)
        model = torch.load('ckpt/{}_finetune_model_820.pkl'.format(finetune_set))
        test_model(model, word_emb_dir, test_dir, finetuneset=finetune_set)
        



