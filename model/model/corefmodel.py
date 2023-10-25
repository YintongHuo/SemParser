from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .charcnn import CharCNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from torch.autograd import Variable



class CoRef(nn.Module):

    def __init__(self, data):
        super(CoRef, self).__init__()

        self.gpu = data.HP_gpu

        self.CharCNN_layer = CharCNN(data.char_alphabet.size(), data.char_emb_dim, data.HP_char_hidden_dim, data.HP_dropout, gpu=False)
        self.WordEmbedding_layer(data)
        self.FeatureEmbedding_layer()

        #lstm
        self.lstm_input_size = data.word_emb_dim + data.char_emb_dim + self.feat_emb_dim
        self.lstm_hidden = 128
        self.lstm_layer = data.HP_lstm_layer
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                            bidirectional=True)
        self.droplstm = nn.Dropout(0.5)

        #mention_rep and score
        self.mention_dim = 100
        self.mention_rep = nn.Linear(self.lstm_hidden * 2, self.mention_dim)

        self.mention_score = nn.Linear(self.lstm_hidden*2, 1)

        #last classifier
        self.last_hidden_dim = 50
        self.second_last_layer = torch.nn.Linear(self.mention_dim * 2 + self.word_emb_dim, self.last_hidden_dim)
        self.last_layer = torch.nn.Linear(self.last_hidden_dim, 1)

        self.to_classify = torch.nn.Linear(self.mention_dim * 2, 2)

        #loss func
        self.loss_func = nn.CrossEntropyLoss()

        self.token_type_classifier = torch.nn.Linear(self.mention_dim, 3)




    # def cross_entropy(self, out, label):


    def WordEmbedding_layer(self, data):
        # self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_emb_dim = data.word_emb_dim
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), data.word_emb_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))

    def FeatureEmbedding_layer(self):
        self.feat_emb_dim = 30
        self.feature_embedding = nn.Linear(16, self.feat_emb_dim)


    def get_words_embeddings(self, word_inputs):
        word_embs = self.word_embedding(word_inputs)


    def WordRep(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, word_features):
            """
                input:
                    word_inputs: (batch_size, sent_len)
                    features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                    word_seq_lengths: list of batch_size, (batch_size,1)
                    char_inputs: (batch_size*sent_len, word_length)
                    char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                    char_seq_recover: variable which records the char order information, used to recover char order
                output:
                    Variable(batch_size, sent_len, hidden_dim)
            """

            batch_size = word_inputs.size(0)
            sent_len = word_inputs.size(1)
            # print(word_inputs.size())
            word_embs = self.word_embedding(word_inputs) # (batch_size, sent_len, 100)

            word_list = [word_embs]
            word_features = self.feature_embedding(word_features)
            # print(word_features.size())
            # print(word_embs.size())

            ## calculate char lstm last hidden
            # print("charinput:", char_inputs)
            # exit(0)
            char_features = self.CharCNN_layer.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover] # (sent_len, 50)
            char_features = char_features.view(batch_size, sent_len, -1) # (batch_size, sent_len, 50)
            ## concat word and char together
            word_list.append(char_features)
            word_embs = torch.cat([word_embs, char_features, word_features], 2) # (batch_size, sent_len, 150)

            # word_embs = torch.cat(word_list, 2)
            # word_represent = self.drop(word_embs)

            return word_embs

    def lstm_rep(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, word_features):

        word_represent = self.WordRep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, word_features) # (batch_size, seq_len, 150)
        # print(word_represent)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out) ## lstm_out (seq_len, batch_size, hidden_size)
        feature_out = self.droplstm(lstm_out) ## feature_out (batch_size, seq_len, hidden_size)
        # print(lstm_out)
        feature_out = feature_out.transpose(1, 0)  ## feature_out (batch_size, seq_len, hidden_size)
        # print(feature_out)

        # outputs = self.hidden2tag(feature_out) The linear layer that maps from hidden state space to tag space

        return feature_out



    def predict(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, combinations, combination_labels, word_features):

        # print(word_inputs)
        # print(char_inputs)
        lstm_feat_out = self.lstm_rep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, word_features) # (batch_size, seq_len, lstm_hidden_size*2)
        mention_feat = self.mention_rep(lstm_feat_out) # (batch_size, seq_len, mention_size)

        batch_size = word_inputs.size(0)
        mention_feat_size = mention_feat.size(2)
        predicts = [] # for each token id, predict its corefence id
        for cur_comb, cur_comb_label in zip(combinations[0], combination_labels[0]):
            feat_combines = mention_feat[:, torch.tensor(cur_comb)].squeeze(0)
            feat_combines = feat_combines.view(batch_size, -1, mention_feat_size*2)

            feat_join = torch.empty([batch_size, len(cur_comb), self.word_emb_dim])
            for idx, pair in enumerate(cur_comb):

                if pair[1]+1 == pair[0]:
                    joint_embedding = self.word_embedding(torch.tensor([[52974]]))
                    # word_idx = word_inputs[:, torch.arange(pair[1], pair[0] + 1)]
                    # joint_embedding = self.word_embedding(word_idx)
                else:
                    word_idx = word_inputs[:, torch.arange(pair[1]+1, pair[0])]
                    joint_embedding = self.word_embedding(word_idx)
                joint_embedding = torch.mean(joint_embedding, 1).squeeze(0)
                feat_join[:, idx, :] = joint_embedding
            feat_combines = torch.cat([feat_combines, feat_join], dim=2)

            preds = self.last_layer(self.second_last_layer(feat_combines))
            scores = preds[:, :, 0].squeeze(0)
            id_pair = cur_comb[torch.argmax(scores)]
            predicts.append(id_pair)

            # print(cur_comb)
            # print(scores)


        predict_token_label = torch.argmax(self.token_type_classifier(mention_feat).squeeze(), dim=1)

        return predicts, predict_token_label


    def compute_single_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                         combination, combination_label, token_label, word_features):

        self.word_embedding.weight.requires_grad = False

        lstm_feat_out = self.lstm_rep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover, word_features)  # (batch_size, seq_len, lstm_hidden_size*2)
        # print(lstm_feat_out)
        mention_feat = self.mention_rep(lstm_feat_out)  # (batch_size, seq_len, mention_size)

        batch_size = word_inputs.size(0)
        mention_feat_size = mention_feat.size(2)
        # combs_for_each_token = []
        # loss = 0
        loss_list = []

        feat_combines = mention_feat[:, torch.tensor(combination)].squeeze(0)
        feat_combines = feat_combines.view(batch_size, -1, mention_feat_size * 2)

        feat_join = torch.empty([batch_size, len(combination), self.word_emb_dim])
        for idx, pair in enumerate(combination):
            if pair[1]+1 == pair[0]:
                word_idx = word_inputs[:, torch.arange(pair[1], pair[0] + 1)]
                # word_idx = torch.range()
                joint_embedding = self.word_embedding(torch.tensor([[52826]]))
                # print(joint_embedding)
            else:
                word_idx = word_inputs[:, torch.arange(pair[1] + 1, pair[0])]
                joint_embedding = self.word_embedding(word_idx)
                # print(joint_embedding)
            joint_embedding = torch.mean(joint_embedding, 1).squeeze(0)
            feat_join[:, idx, :] = joint_embedding
        feat_combines = torch.cat([feat_combines, feat_join], dim=2)

        preds = self.last_layer(self.second_last_layer(feat_combines))
        scores = preds[:, :, 0]
        gold_label = torch.tensor([combination_label.index(1)])

        mention_pair_loss = self.loss_func(scores, gold_label)


        token_class_feat = self.token_type_classifier(mention_feat).squeeze()

        token_label = token_label.squeeze()
        #token_label: [0,0,0,0,1]
        # print(token_label)
        # print(token_class_feat)

        token_type_loss = self.loss_func(token_class_feat, token_label) / 7


        loss = mention_pair_loss + token_type_loss

        return loss




