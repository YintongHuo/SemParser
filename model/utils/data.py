from __future__ import print_function
from __future__ import absolute_import
import sys
from .alphabet import Alphabet
from .functions import *

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"


class Data:
    def __init__(self, train_dir, word_emb_dir):
        self.sentence_classification = False
        self.MAX_SENTENCE_LENGTH = 1000
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = True
        self.norm_char_emb = True
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')


        self.split_token = ' ||| '
        # self.seg = True

        ### I/O
        self.train_dir = train_dir
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None

        self.decode_dir = None
        self.dset_dir = None  ## data vocabulary related file
        self.model_dir = None  ## model save file
        self.load_model_dir = None  ## model load file

        self.word_emb_dir = word_emb_dir
        self.char_emb_dir = None
        self.feature_emb_dirs = []

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.feature_alphabet_sizes = []
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 100
        self.char_emb_dim = 30

        ###Networks
        self.word_feature_extractor = "LSTM"  ## "LSTM"/"CNN"/"GRU"/
        self.use_char = True
        self.char_feature_extractor = "CNN"  ## "LSTM"/"CNN"/"GRU"/None
        # self.use_crf = True
        # self.nbest = None

        ## Training
        self.average_batch_loss = False
        self.optimizer = "SGD"  ## "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"
        self.status = "train"
        ### Hyperparameters
        # self.HP_cnn_layer = 4
        self.HP_iteration = 30
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 30
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 2
        self.HP_bilstm = True

        self.HP_gpu = False
        self.HP_lr = 0.0001
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8


    def build_alphabet(self, input_file, embedding_path):
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if line != '\n':
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)


        # add embeds
        with open(embedding_path, 'r', encoding="utf8") as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if sys.version_info[0] < 3:
                    first_col = tokens[0].decode('utf-8')
                else:
                    first_col = tokens[0]
                self.word_alphabet.add(first_col)
                for char in first_col:
                    self.char_alphabet.add(char)

        # Add other tokens?
        self.word_alphabet.add('[EPT]')
        for char in '[EPT]':
            self.char_alphabet.add(char)

        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrained word embedding, norm: %s, dir: %s" % (self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir,
                                                                                       self.word_alphabet,
                                                                                       self.word_emb_dim,
                                                                                       self.norm_word_emb)
        if self.char_emb_dir:
            print("Load pretrained char embedding, norm: %s, dir: %s" % (self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir,
                                                                                       self.char_alphabet,
                                                                                       self.char_emb_dim,
                                                                                       self.norm_char_emb)

    def generate_instance(self, name):
        self.fix_alphabet()
        if name == "train":
            self.word_alphabet.save('../data', 'WordAlphabet')
            self.char_alphabet.save('../data', 'CharAlphabet')
            self.train_texts, self.train_Ids = read_V1_instance(self.train_dir, self.word_alphabet, self.char_alphabet,
                                                             self.MAX_SENTENCE_LENGTH)
        # elif name == "dev":
        #     self.dev_texts, self.dev_Ids = read_V1_instance(self.dev_dir, self.word_alphabet, self.char_alphabet,
        #                                                      self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.word_alphabet.load('../data', 'WordAlphabet')
            self.char_alphabet.load('../data', 'CharAlphabet')
            self.test_texts, self.test_Ids = read_V1_instance(self.test_dir, self.word_alphabet, self.char_alphabet,
                                                             self.MAX_SENTENCE_LENGTH, if_train=False)
            self.word_alphabet.close()

        # elif name == "dataset":
        #     self.word_alphabet.load('data', 'WordAlphabet')
        #     self.char_alphabet.load('data', 'CharAlphabet')
        #     self.test_texts, self.test_Ids = read_instance_from_tokens(self.data_tokens, self.word_alphabet, self.char_alphabet,
        #                                                      self.MAX_SENTENCE_LENGTH, if_train=False)
        #     self.word_alphabet.close()

        # elif name == "raw":
        #     self.word_alphabet.load('data', 'WordAlphabet')
        #     self.char_alphabet.load('data', 'CharAlphabet')
        #     self.fix_alphabet()
        #     self.train_texts, self.train_Ids = read_V1_instance(self.train_dir, self.word_alphabet, self.char_alphabet,
        #                                                      self.MAX_SENTENCE_LENGTH, if_train=False)
        elif name == 'finetune':
            self.word_alphabet.load('../data', 'WordAlphabet')
            self.char_alphabet.load('../data', 'CharAlphabet')
            self.train_texts, self.train_Ids = read_V1_instance(self.train_dir, self.word_alphabet, self.char_alphabet,
                                                             self.MAX_SENTENCE_LENGTH)

        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def write_decoded_results(self, predict_results, name):

        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        fout = open(self.decode_dir, 'w')
        for idx in range(sent_num):
            if self.sentence_classification:
                fout.write(" ".join(content_list[idx][0]) + "\t" + predict_results[idx] + '\n')
            else:
                sent_length = len(predict_results[idx])
                for idy in range(sent_length):
                    ## content_list[idx] is a list with [word, char, label]
                    fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
                fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, self.decode_dir))

    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def write_nbest_decoded_results(self, predict_results, pred_scores, name):
        ## predict_results : [whole_sent_num, nbest, each_sent_length]
        ## pred_scores: [whole_sent_num, nbest]
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        assert (sent_num == len(pred_scores))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx][0])
            nbest = len(predict_results[idx])
            score_string = "# "
            for idz in range(nbest):
                score_string += format(pred_scores[idx][idz], '.4f') + " "
            fout.write(score_string.strip() + "\n")

            for idy in range(sent_length):
                try:  # Will fail with python3
                    label_string = content_list[idx][0][idy].encode('utf-8') + " "
                except:
                    label_string = content_list[idx][0][idy] + " "
                for idz in range(nbest):
                    label_string += predict_results[idx][idz][idy] + " "
                label_string = label_string.strip() + "\n"
                fout.write(label_string)
            fout.write('\n')
        fout.close()
        print("Predict %s %s-best result has been written into file. %s" % (name, nbest, self.decode_dir))


def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False