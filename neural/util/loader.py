from __future__ import print_function
import os
import torch

import copy

import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from .utils import *
import codecs
import pickle
import itertools
from collections import Counter


class Loader(object):

    def __init__(self):
        pass

    def word_mapping(self, dataset):

        p = [k[0] + ' ' + k[1] for k in dataset]
        words = [[x.lower() for x in s.split()] for s in p]

        # Create word list
        dico = create_dico(words)

        dico['<PAD>'] = 10000001
        dico['<UNK>'] = 10000000

        # Keep only words that appear more than 1
        dico = {k: v for k, v in dico.items() if v >= 2}

        word_to_id, id_to_word = create_mapping(dico)

        print("Found %i unique words (%i in total)" % (
            len(dico), sum(len(x) for x in words)
        ))
        return dico, word_to_id, id_to_word

    def load_yahoo(self, datapath, pretrained, word_dim=100, answer_count = 5):

        # trainpath = os.path.join(datapath, 'train.txt')
        # testpath = os.path.join(datapath, 'test.txt')
        #
        # train_data = []
        # with open(trainpath) as f:
        #     for line in f:
        #         content = line.strip().split('#')
        #         if len(content) != 3:
        #             continue
        #         if int(content[2]) != 0 and int(content[2]) != 1:
        #             continue
        #         train_data.append((content[0], content[1], content[2]))
        #
        # test_data = []
        # with open(testpath) as f:
        #     for line in f:
        #         content = line.strip().split('#')
        #         if len(content) != 3:
        #             continue
        #         if int(content[2]) != 0 and int(content[2]) != 1:
        #             print(content[2])
        #             continue
        #         test_data.append((content[0], content[1], content[2]))

        trainpath = os.path.join(datapath, 'data.txt')

        data = []
        with open(trainpath) as f:
            for line in f:
                content = line.strip().split('#')
                if len(content) != 3:
                    continue
                if int(content[2]) != 0 and int(content[2]) != 1:
                    continue
                data.append((content[0], content[1], content[2]))

        # data_index = np.random.RandomState(seed=0).permutation(int(len(data) / answer_count))
        data_index = np.arange(int(len(data) / answer_count))
        split = 1600

        train_data = [data[no] for sampleID in data_index[:split]
                          for no in range(sampleID * answer_count, sampleID * answer_count + 5)]
        test_data = [data[no] for sampleID in data_index[split:]
                          for no in range(sampleID * answer_count, sampleID * answer_count + 5)]

        dico_words_train = self.word_mapping(train_data)[0]

        all_embedding = False

        dico_words, word_to_id, id_to_word = augment_with_pretrained(
            dico_words_train.copy(),
            pretrained,
            list(itertools.chain.from_iterable(
                [[w.lower() for w in (s[0] + ' ' + s[1]).split()] for s in test_data]  # 样本包含问题和答案
            )
            ) if not all_embedding else None)

        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_data)

        train_data_final = prepare_dataset(train_data, word_to_id, tag_to_id)
        test_data_final = prepare_dataset(test_data, word_to_id, tag_to_id)

        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

        # If the words in the training set do not exist in the pre-trained word vector,
        # replace them with a randomly generated uniformly distributed vector
        not_exist = 0
        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]
            else:
                not_exist += 1
        #print(" %d new words" % (not_exist))

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }

        # npr = np.random.RandomState(seed=0)
        # data_index = npr.permutation(int(len(test_data_final) / answer_count))
        #
        # test_data_final = [test_data_final[no] for sampleID in data_index
        #                   for no in range(sampleID * answer_count, sampleID * answer_count + 5)]

        return train_data_final, test_data_final, mappings
