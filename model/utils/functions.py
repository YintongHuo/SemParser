from __future__ import print_function
from __future__ import absolute_import
import sys
import itertools
import numpy as np
import re

def StartsWithUppercaseFeature(token):
    if int(token[:1].istitle()):
        return [0,1]
    else:
        return [1,0]

def TokenLengthFeature(token):
    if len(token) >10:
        return [0,1]
    else:
        return [1,0]

def ContainsDigitsFeature(token):
    regexp_contains_digits = re.compile(r'[0-9]+')
    if regexp_contains_digits.search(token):
        return [0,1]
    else:
        return [1,0]

def ContainsPunctuationFeature(token):
    regexp_contains_punctuation = re.compile(r'[\.\,\:\;\(\)\[\]\?\!]+')
    if regexp_contains_punctuation.search(token):
        return [0,1]
    else:
        return [1,0]

def OnlyDigitsFeature(token):
    regexp_contains_only_digits = re.compile(r'^[0-9]+$')
    if regexp_contains_only_digits.search(token):
        return [0,1]
    else:
        return [1,0]

def OnlyPunctuationFeature(token):
    regexp_contains_only_punctuation = re.compile(r'^[\.\,\:\;\(\)\[\]\?\!]+$')
    if regexp_contains_only_punctuation.search(token):
        return [0,1]
    else:
        return [1,0]

def CamelFeature(token):
    camel = re.compile('^[a-z]+(?:[A-Z][a-z]+)*$')
    if camel.search(token):
        return [0,1]
    else:
        return [1,0]

def PascalFeature(token):
    pascal = re.compile('^[A-Z][a-z]+(?:[A-Z][a-z]+)*$')
    if pascal.search(token):
        return [0,1]
    else:
        return [1,0]

def extract_feature(token):
    swu = StartsWithUppercaseFeature(token)
    tlf = TokenLengthFeature(token)
    cdf = ContainsDigitsFeature(token)
    cpf = ContainsPunctuationFeature(token)
    odf = OnlyDigitsFeature(token)
    opf = OnlyPunctuationFeature(token)
    cf = CamelFeature(token)
    pf = PascalFeature(token)

    features = list()
    features.extend(swu)
    features.extend(tlf)
    features.extend(cdf)
    features.extend(cpf)
    features.extend(odf)
    features.extend(opf)
    features.extend(cf)
    features.extend(pf)

    return features

def normalize_number(word):
    new_word = ""
    if re.search('^(req-)?[a-f0-9x\-\.]+$', word):
        new_word = '000'
        # new_word = '0000'
        # for char in word:
        #     new_word += '0' if char != '-' else char
    else:
        new_word = word
    return new_word


def normalize_word(word):
    word = normalize_number(word)
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    # new_word = re.sub('0+','0', new_word)
    return new_word

def exist_positive(combination_labels):
    for comb in combination_labels:
        comb = comb[1:]
        try:
            comb.index(1)
            return True
        except:
            pass
    return False

def normalize_url(word):
    splits = re.split('(/)', word)
    if len(splits) > 2:
        return ''.join(splits[:4])
    else:
        return word

def read_instance(input_file, word_alphabet, char_alphabet, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):

    in_lines = open(input_file,'r', encoding="utf8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    combinations = []
    combination_labels = []
    word_Ids = []
    char_Ids = []


    for line in in_lines:
        if line != '\n':
            pairs = line.strip().split()
            word = pairs[0]
            if sys.version_info[0] < 3:
                word = word.decode('utf-8')
            words.append(word)
            # word = normalize_word(word)
            if len(pairs) > 1:
                label = str(pairs[-1])
            else:
                label = '0'
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))

            ## get char
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
                # read label
                # combinations = itertools.combinations(range(len(labels)), 2)
                combinations = []
                combination_labels = []
                for i in range(2, len(labels)):
                    tmp_combinations = []
                    tmp_combination_labels = []
                    coref_flag = 0
                    for j in range(0, i):
                        tmp_combinations.append([i, j])
                        if coref_flag == 1:
                            tmp_combination_labels.append(0)
                        elif labels[i] != '0' and labels[j] != '0' and labels[i] == labels[j]:
                            tmp_combination_labels.append(1)
                            coref_flag = 1
                            # elif j == 0 and coref_flag == 0:
                            #     tmp_combination_labels.append(1)
                        else:
                            tmp_combination_labels.append(0)
                    try:
                        tmp_combination_labels.index(1)
                    except:
                        tmp_combination_labels[0] = 1
                    combinations.append(tmp_combinations)
                    combination_labels.append(tmp_combination_labels)

                if exist_positive(combination_labels):
                    instence_texts.append([words, chars, combinations, combination_labels])
                    instence_Ids.append([word_Ids, char_Ids, combinations, combination_labels])
            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            combinations = []
            combination_labels = []

    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
        combinations = []
        combination_labels = []
        for i in range(2, len(labels)):
            tmp_combinations = []
            tmp_combination_labels = []
            coref_flag = 0
            for j in range(0, i):
                tmp_combinations.append([i, j])
                if coref_flag == 1:
                    tmp_combination_labels.append(0)
                elif labels[i] != '0' and labels[j] != '0' and labels[i] == labels[j]:
                    tmp_combination_labels.append(1)
                    coref_flag = 1
                # elif j == 0 and coref_flag == 0:
                #     tmp_combination_labels.append(1)
                else:
                    tmp_combination_labels.append(0)
            try:
                tmp_combination_labels.index(1)
            except:
                tmp_combination_labels[0] = 1
            combinations.append(tmp_combinations)
            combination_labels.append(tmp_combination_labels)

        if exist_positive(combination_labels):
            instence_texts.append([words, chars, combinations, combination_labels])
            instence_Ids.append([word_Ids, char_Ids, combinations, combination_labels])
        words = []
        chars = []
        labels = []
        word_Ids = []
        char_Ids = []
        combinations = []
        combination_labels = []

    # print(instence_texts)
    print('We have {} data in total'.format(len(instence_Ids)))
    return instence_texts, instence_Ids



def read_V1_instance(input_file, word_alphabet, char_alphabet, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>', if_train=True):

    in_lines = open(input_file,'r', encoding="utf8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    combinations = []
    combination_labels = []
    token_labels_Ids = []
    token_labels = []
    word_Ids = []
    char_Ids = []
    word_features = []


    for line in in_lines:
        if line != '\n':
            pairs = line.strip().split()
            word = pairs[0]
            word_features.append(extract_feature(word))
            if sys.version_info[0] < 3:
                word = word.decode('utf-8')
            words.append(word)
            word = normalize_word(word)
            word = normalize_url(word)
            if len(pairs) > 1:
                cur_label = str(pairs[-1])
                # label = str(pairs[-1])
                #determine label: [0:Null, 1: MA, 2:MI], remove the 'MA'/'MI'
                if 'MA' in cur_label:
                    token_label_Id = 1
                    token_label = 'MA'
                    if 'MA' != cur_label:
                        label = cur_label[2:]
                    else:
                        label = '0'
                if 'MI' in cur_label:
                    token_label_Id = 2
                    token_label = 'MI'
                    if 'MI' != cur_label:
                        label = cur_label[2:]
                    else:
                        label = '0'
            else:
                token_label_Id = 0
                token_label = 'NULL'
                label = '0'
            labels.append(label)
            token_labels_Ids.append(token_label_Id)
            token_labels.append(token_label)
            word_Ids.append(word_alphabet.get_index(word))


            ## get char
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)


        else:
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
                # print(words)
                # print(word_Ids)
                # read label
                # combinations = itertools.combinations(range(len(labels)), 2)
                combinations = []
                combination_labels = []
                for i in range(2, len(labels)):
                    tmp_combinations = []
                    tmp_combination_labels = []
                    coref_flag = 0
                    for j in range(0, i):
                        tmp_combinations.append([i, j])
                        if coref_flag == 1:
                            tmp_combination_labels.append(0)
                        elif labels[i] != '0' and labels[j] != '0' and labels[i] == labels[j]:
                            tmp_combination_labels.append(1)
                            coref_flag = 1
                            # elif j == 0 and coref_flag == 0:
                            #     tmp_combination_labels.append(1)
                        else:
                            tmp_combination_labels.append(0)
                    try:
                        tmp_combination_labels.index(1)
                    except:
                        tmp_combination_labels[0] = 1
                    combinations.append(tmp_combinations)
                    combination_labels.append(tmp_combination_labels)

                # if if_train and exist_positive(combination_labels):
                #     instence_texts.append([words, chars, combinations, combination_labels, token_labels])
                #     instence_Ids.append([word_Ids, char_Ids, combinations, combination_labels, token_labels_Ids])
                # elif not if_train:
                instence_texts.append([words, chars, combinations, combination_labels, token_labels, word_features])
                instence_Ids.append([word_Ids, char_Ids, combinations, combination_labels, token_labels_Ids, word_features])
                # else:
                #     print(words)
                #     print(labels)
            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            token_labels = []
            token_labels_Ids = []
            combinations = []
            combination_labels = []
            word_features = []

    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
        combinations = []
        combination_labels = []
        for i in range(2, len(labels)):
            tmp_combinations = []
            tmp_combination_labels = []
            coref_flag = 0
            for j in range(0, i):
                tmp_combinations.append([i, j])
                if coref_flag == 1:
                    tmp_combination_labels.append(0)
                elif labels[i] != '0' and labels[j] != '0' and labels[i] == labels[j]:
                    tmp_combination_labels.append(1)
                    coref_flag = 1
                # elif j == 0 and coref_flag == 0:
                #     tmp_combination_labels.append(1)
                else:
                    tmp_combination_labels.append(0)
            try:
                tmp_combination_labels.index(1)
            except:
                tmp_combination_labels[0] = 1
            combinations.append(tmp_combinations)
            combination_labels.append(tmp_combination_labels)

        # empty_token = word_alphabet.get_index('[EPT]')
        # print(empty_token)
        # if if_train and exist_positive(combination_labels):
        #     instence_texts.append([words, chars, combinations, combination_labels, token_labels])
        #     instence_Ids.append([word_Ids, char_Ids, combinations, combination_labels, token_labels_Ids])
        # elif not if_train:
        instence_texts.append([words, chars, combinations, combination_labels, token_labels, word_features])
        instence_Ids.append([word_Ids, char_Ids, combinations, combination_labels, token_labels_Ids, word_features])
        # else:
        #     print(words)
        #     print(labels)
        words = []
        chars = []
        labels = []
        word_Ids = []
        char_Ids = []
        token_labels = []
        token_labels_Ids = []
        combinations = []
        combination_labels = []
        word_features = []

    # print(instence_texts)
    print('We have {} data in total in V1'.format(len(instence_Ids)))
    return instence_texts, instence_Ids


def read_instance_from_tokens(data_tokens, word_alphabet, char_alphabet, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>', if_train=True):
    # [[t_1,t_2], [], ...]
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    combinations = []
    combination_labels = []
    token_labels_Ids = []
    token_labels = []
    word_Ids = []
    char_Ids = []
    word_features = []

    for tokens in data_tokens:
        for token in tokens:
            pairs = token.strip().split()
            word = pairs[0]
            word_features.append(extract_feature(word))
            if sys.version_info[0] < 3:
                word = word.decode('utf-8')
            words.append(word)
            word = normalize_url(word)
            word = normalize_word(word)
            if len(pairs) > 1:
                cur_label = str(pairs[-1])
                # label = str(pairs[-1])
                # determine label: [0:Null, 1: MA, 2:MI], remove the 'MA'/'MI'
                if 'MA' in cur_label:
                    token_label_Id = 1
                    token_label = 'MA'
                    if 'MA' != cur_label:
                        label = cur_label[2:]
                    else:
                        label = '0'
                if 'MI' in cur_label:
                    token_label_Id = 2
                    token_label = 'MI'
                    if 'MI' != cur_label:
                        label = cur_label[2:]
                    else:
                        label = '0'
            else:
                token_label_Id = 0
                token_label = 'NULL'
                label = '0'
            labels.append(label)
            token_labels_Ids.append(token_label_Id)
            token_labels.append(token_label)
            word_Ids.append(word_alphabet.get_index(word))

            ## get char
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                # read label
                # combinations = itertools.combinations(range(len(labels)), 2)
                combinations = []
                combination_labels = []
                for i in range(2, len(labels)):
                    tmp_combinations = []
                    tmp_combination_labels = []
                    coref_flag = 0
                    for j in range(0, i):
                        tmp_combinations.append([i, j])
                        if coref_flag == 1:
                            tmp_combination_labels.append(0)
                        elif labels[i] != '0' and labels[j] != '0' and labels[i] == labels[j]:
                            tmp_combination_labels.append(1)
                            coref_flag = 1
                            # elif j == 0 and coref_flag == 0:
                            #     tmp_combination_labels.append(1)
                        else:
                            tmp_combination_labels.append(0)
                    try:
                        tmp_combination_labels.index(1)
                    except:
                        tmp_combination_labels[0] = 1
                    combinations.append(tmp_combinations)
                    combination_labels.append(tmp_combination_labels)

                instence_texts.append([words, chars, combinations, combination_labels, token_labels, word_features])
                instence_Ids.append(
                    [word_Ids, char_Ids, combinations, combination_labels, token_labels_Ids, word_features])

            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            token_labels = []
            token_labels_Ids = []
            combinations = []
            combination_labels = []
            word_features = []

    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
        combinations = []
        combination_labels = []
        for i in range(2, len(labels)):
            tmp_combinations = []
            tmp_combination_labels = []
            coref_flag = 0
            for j in range(0, i):
                tmp_combinations.append([i, j])
                if coref_flag == 1:
                    tmp_combination_labels.append(0)
                elif labels[i] != '0' and labels[j] != '0' and labels[i] == labels[j]:
                    tmp_combination_labels.append(1)
                    coref_flag = 1
                # elif j == 0 and coref_flag == 0:
                #     tmp_combination_labels.append(1)
                else:
                    tmp_combination_labels.append(0)
            try:
                tmp_combination_labels.index(1)
            except:
                tmp_combination_labels[0] = 1
            combinations.append(tmp_combinations)
            combination_labels.append(tmp_combination_labels)


        instence_texts.append([words, chars, combinations, combination_labels, token_labels, word_features])
        instence_Ids.append([word_Ids, char_Ids, combinations, combination_labels, token_labels_Ids, word_features])

        words = []
        chars = []
        labels = []
        word_Ids = []
        char_Ids = []
        token_labels = []
        token_labels_Ids = []
        combinations = []
        combination_labels = []
        word_features = []

    # print(instence_texts)
    print('We have {} data in total'.format(len(instence_Ids)))
    return instence_texts, instence_Ids



def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            # print(word)
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            elif embedd_dim + 1 != len(tokens):
                ## ignore illegal embedding line
                continue
                # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd

    # add two special token [TMP]
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_dict['[TMP]'] = np.random.uniform(-scale, scale, [1, embedd_dim])
    embedd_dict['[STRUCT]'] = np.random.uniform(-scale, scale, [1, embedd_dim])
    # embedd_dict['[REP]'] = np.random.uniform(-scale, scale, [1, embedd_dim])
    # print(embedd_dict['[TMP]'])
    # print(embedd_dict['request'])
    return embedd_dict, embedd_dim

if __name__ == '__main__':

    embedd_dict, embedd_dim = load_pretrain_emb('../data/w2v.emb')
    print(list(embedd_dict.items())[:3])
    print(embedd_dim)