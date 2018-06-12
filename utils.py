from dataset.dataset import *
import os.path as osp
import pandas as pd
from tqdm import *
import numpy as np
import torch


def np_to_var(np_arr, requires_grad=True):
    return torch.autograd.Variable(torch.from_numpy(np_arr), requires_grad=requires_grad)


class CsvAvitoProvider(DefaultClassProvider):
    def __init__(self, csv_path, images_path):
        self.objs = []
        self.lbls = []

        csv = pd.read_csv(csv_path)

        for index, r in tqdm(csv.iterrows()):
            if len(str(r['image'])) < 4:
                continue

            if r['deal_probability'] < 0.2:
                continue

            if not osp.exists(osp.join(images_path, r['image'] + '.jpg')):
                continue

            self.objs.append(osp.join(images_path, r['image'] + '.jpg'))
            self.lbls.append(r['deal_probability'])

    def __getitem__(self, item):
        return self.objs[item], self.lbls[item]

    def __len__(self):
        return len(self.objs)
