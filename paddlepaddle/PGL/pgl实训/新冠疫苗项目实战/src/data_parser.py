#-*- coding: utf-8 -*-
import os
import sys
sys.path.append("../")
import copy
import random
import argparse
import logging
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict, namedtuple

from propeller import log
log.setLevel(logging.DEBUG)

class GraphParser(object):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode

    def parse(self, sample):
        labels = []
        nfeat = []
        efeat = []
        edges = []
        train_mask = []
        test_mask = []

        sequence = sample['sequence']
        predicted_loop_type = sample['predicted_loop_type']
        seq_length = sample['seq_length']
        seq_scored = sample['seq_scored']

        pair_info = match_pair(sample['structure'])

        paired_nodes = {}
        for j in range(seq_length):
            add_base_node(nfeat, sequence[j], predicted_loop_type[j])

            if j + 1 < seq_length: # edge between current node and next node
                add_edges_between_base_nodes(edges, efeat, j, j + 1)

            if pair_info[j] != -1:
                if pair_info[j] not in paired_nodes:
                    paired_nodes[pair_info[j]] = [j]
                else:
                    paired_nodes[pair_info[j]].append(j)

            train_mask.append(j < seq_scored)
            test_mask.append(True)

        if self.config.add_edge_for_paired_nodes:
            for pair in paired_nodes.values():
                add_edges_between_paired_nodes(edges, efeat, pair[0], pair[1])

        if self.config.add_codon_nodes:
            codon_node_idx = seq_length - 1
            for j in range(seq_length):
                if j % 3 == 0:
                    # add codon node
                    add_codon_node(nfeat)
                    codon_node_idx += 1
                    train_mask.append(False)
                    test_mask.append(False)
                    if self.mode != "test":
                        labels.append([0, 0, 0])

                    if codon_node_idx > seq_length:
                        # add edges between adjacent codon nodes
                        add_edges_between_codon_nodes(
                                edges, efeat, codon_node_idx - 1, codon_node_idx)

                # add edges between codon node and base node
                add_edges_between_codon_and_base_node(
                        edges, efeat, j, codon_node_idx)

        if self.mode != 'test':
            react = sample['reactivity']
            deg_Mg_pH10 = sample['deg_Mg_pH10']
            deg_Mg_50C = sample['deg_Mg_50C']

            for j in range(seq_length):
                if j < seq_scored:
                    labels.append([react[j], deg_Mg_pH10[j], deg_Mg_50C[j] ])
                else:
                    labels.append([0, 0, 0])

        gdata = {}
        gdata['nfeat'] = np.array(nfeat, dtype="float32")
        gdata['edges'] = np.array(edges, dtype="int64")
        gdata['efeat'] = np.array(efeat, dtype="float32")
        if self.mode != "test":
            gdata['labels'] = np.array(labels, dtype="float32")
            gdata['mask'] = np.array(train_mask, dtype=bool).reshape(-1, 1)
        else:
            # fake labels
            gdata['labels'] = np.zeros((self.config.batch_size, self.config.num_class))
            gdata['mask'] = np.array(test_mask, dtype=bool).reshape(-1, 1)

        return gdata

def add_node(node_features, feature):
    node_features.append(feature)

def add_base_node(node_features, sequence, predicted_loop_type):
    feature = [
        0, # is codon node
        sequence == 'A',
        sequence == 'C',
        sequence == 'G',
        sequence == 'U',
        predicted_loop_type == 'S',
        predicted_loop_type == 'M',
        predicted_loop_type == 'I',
        predicted_loop_type == 'B',
        predicted_loop_type == 'H',
        predicted_loop_type == 'E',
        predicted_loop_type == 'X',
        0.0, # bpps_sum,
        0.0, # bpps_nb,
    ]
    add_node(node_features, feature)

def add_codon_node(node_features):
    feature = [
        1, # is codon node
        0, # sequence == 'A',
        0, # sequence == 'C',
        0, # sequence == 'G',
        0, # sequence == 'U',
        0, # predicted_loop_type == 'S',
        0, # predicted_loop_type == 'M',
        0, # predicted_loop_type == 'I',
        0, # predicted_loop_type == 'B',
        0, # predicted_loop_type == 'H',
        0, # predicted_loop_type == 'E',
        0, # predicted_loop_type == 'X',
        0, # bpps_sum
        0, # bpps_nb
    ]
    add_node(node_features, feature)

# add directed edge for node1 -> node2 and for node2 -> node1
def add_edges(edge_index, edge_features, node1, node2, feature1, feature2):
    edge_index.append([node1, node2])
    edge_features.append(feature1)
    edge_index.append([node2, node1])
    edge_features.append(feature2)

def add_edges_between_base_nodes(edge_index, edge_features, node1, node2):
    edge_feature1 = [
        0, # is edge for paired nodes
        0, # is edge between codon node and base node
        0, # is edge between coden nodes
        1, # forward edge: 1, backward edge: -1
        1, # bpps if edge is for paired nodes
    ]
    edge_feature2 = [
        0, # is edge for paired nodes
        0, # is edge between codon node and base node
        0, # is edge between coden nodes
        -1, # forward edge: 1, backward edge: -1
        1, # bpps if edge is for paired nodes
    ]
    add_edges(edge_index, edge_features, node1, node2,
              edge_feature1, edge_feature2)

def add_edges_between_paired_nodes(edge_index, edge_features, node1, node2):
    edge_feature1 = [
        1, # is edge for paired nodes
        0, # is edge between codon node and base node
        0, # is edge between coden nodes
        0, # forward edge: 1, backward edge: -1
        0, # bpps_value, # bpps if edge is for paired nodes
    ]
    edge_feature2 = [
        1, # is edge for paired nodes
        0, # is edge between codon node and base node
        0, # is edge between coden nodes
        0, # forward edge: 1, backward edge: -1
        0, # bpps_value, # bpps if edge is for paired nodes
    ]
    add_edges(edge_index, edge_features, node1, node2,
              edge_feature1, edge_feature2)


def add_edges_between_codon_and_base_node(edge_index, edge_features, node1, node2):
    edge_feature1 = [
        0, # is edge for paired nodes
        1, # is edge between codon node and base node
        0, # is edge between coden nodes
        0, # forward edge: 1, backward edge: -1
        0, # bpps if edge is for paired nodes
    ]
    edge_feature2 = [
        0, # is edge for paired nodes
        1, # is edge between codon node and base node
        0, # is edge between coden nodes
        0, # forward edge: 1, backward edge: -1
        0, # bpps if edge is for paired nodes
    ]
    add_edges(edge_index, edge_features, node1, node2,
              edge_feature1, edge_feature2)

def add_edges_between_codon_nodes(edge_index, edge_features, node1, node2):
    edge_feature1 = [
        0, # is edge for paired nodes
        0, # is edge between codon node and base node
        1, # is edge between coden nodes
        1, # forward edge: 1, backward edge: -1
        0, # bpps if edge is for paired nodes
    ]
    edge_feature2 = [
        0, # is edge for paired nodes
        0, # is edge between codon node and base node
        1, # is edge between coden nodes
        -1, # forward edge: 1, backward edge: -1
        0, # bpps if edge is for paired nodes
    ]
    add_edges(edge_index, edge_features, node1, node2,
              edge_feature1, edge_feature2)



def match_pair(structure):
    pair = [-1] * len(structure)
    pair_no = -1

    pair_no_stack = []
    for i, c in enumerate(structure):
        if c == '(':
            pair_no += 1
            pair[i] = pair_no
            pair_no_stack.append(pair_no)
        elif c == ')':
            pair[i] = pair_no_stack.pop()
    return pair




























































