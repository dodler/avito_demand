from __future__ import print_function
from __future__ import division
from future import standard_library
import sys
import requests
import torch.nn.functional as F
import pickle
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim
from tqdm import *

from lstm_rusvectores.text_utils import tag_mystem, mystem2upos

TRAIN_DATA = '/home/artemlyan/data/avito_demand/train.csv'

training = pd.read_csv(TRAIN_DATA, index_col="item_id", parse_dates=["activation_date"])


def load_w2v_model(file_name: str) -> None:
    print("loading w2v_model...")
    return KeyedVectors.load_word2vec_format(file_name, binary=True, encoding='utf-8')


model = load_w2v_model('/home/artemlyan/models/araneum_upos_skipgram_600_2_2017.bin.gz')
words = list(model.wv.vocab.keys())
words2idx = {}
for i, w in enumerate(words):
    words2idx[w] = i

y = []
tags = []

for index, r in tqdm(training.iterrows()):
    y.append(r['deal_probability'])
    tagged = tag_mystem(r['description'], mapping=mystem2upos)
    tags.append(tagged)


class MyLSTM(nn.Module):

    def __init__(self):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(600, 200, 1, batch_first=True)
        self.linear = nn.Linear(200, 1)

    def forward(self, embedding):
        output, (h_n, c_n) = self.lstm(embedding)
        return self.linear(h_n)


embed = nn.Embedding(len(model.wv.vocab), 600)
weights = torch.FloatTensor(model.syn0)
embed.weight = nn.Parameter(weights)

my_lstm = MyLSTM()
my_lstm.cuda()
my_lstm.train()
my_lstm.float()

optimizer = optim.SGD(lr=1e-3, params=my_lstm.parameters())
criterion = lambda x, y: torch.sqrt(nn.MSELoss().cuda()(x, y))