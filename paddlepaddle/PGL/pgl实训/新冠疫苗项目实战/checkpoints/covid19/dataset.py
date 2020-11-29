#-*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pandas as pd
import glob
from collections import OrderedDict, namedtuple

import pgl
from pgl.utils.data.dataset import Dataset, StreamDataset, HadoopDataset
from propeller import log

from data_parser import GraphParser

def random_split(dataset, frac_train=1.0):
    length = len(dataset)
    perm = list(range(length))
    np.random.shuffle(perm)
    num_train = int(length * frac_train)

    train_indices = perm[0:num_train]
    valid_indices = perm[num_train:]
    assert (len(train_indices) + len(valid_indices)) == length

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    return train_dataset, valid_dataset

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class CovidDataset(Dataset):
    def __init__(self, data_file, config=None, mode='train'):
        self.data_file = data_file
        self.config = config
        self.mode = mode

        df = pd.read_json(self.data_file, lines=True)
        if self.config.shuffle:
            df = df.sample(frac = 1)
            df = df.reset_index(drop=True)

        self.parser = GraphParser(self.config, self.mode)

        # preprocess data
        data = []
        for i in range(len(df)):
            sample = df.loc[i]
            data.append(self.parser.parse(sample))
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class CollateFn(object):
    def __init__(self, config=None, mode='train'):
        self.config = config
        self.mode = mode

    def __call__(self, batch_data):
        num_nodes = []
        num_edges = []
        edges = []
        labels = []
        node_feats = []
        edge_feats = []
        mask = []
        weights = []

        for gdata in batch_data:
            num_nodes.append(len(gdata['nfeat']))
            num_edges.append(len(gdata['edges']))
            edges.append(gdata['edges'])
            labels.append(gdata['labels'])
            node_feats.append(gdata['nfeat'])
            edge_feats.append(gdata['efeat'])
            mask.append(gdata['mask'])

        feed_dict = {}
        feed_dict['num_nodes'] = np.array(num_nodes, dtype="int64")
        feed_dict['num_edges'] = np.array(num_edges, dtype="int64")
        feed_dict['edges'] = np.concatenate(edges)
        feed_dict['node_feat'] = np.concatenate(node_feats)
        feed_dict['edge_feat'] = np.concatenate(edge_feats)
        feed_dict['mask'] = np.concatenate(mask).reshape(-1, )
        if self.mode != 'test':
            feed_dict['labels'] = np.concatenate(labels)

        return feed_dict




































































































