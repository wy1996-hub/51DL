#-*- coding: utf-8 -*-
import sys
sys.path.append("../")
import os
import argparse
import traceback
import re
import io
import json
import yaml
import time 
import logging
from tqdm import tqdm
import numpy as np
from collections import namedtuple

import pgl
from pgl.utils import paddle_helper
from pgl.graph_wrapper import BatchGraphWrapper
from propeller import log
import propeller.paddle as propeller
import paddle.fluid as F
import paddle.fluid.layers as L

from utils.config import prepare_config, make_dir
import layers as GNNlayers

class GNNModel(propeller.train.Model):
    def __init__(self, hparam, mode, run_config):
        self.hparam = hparam
        self.mode = mode
        self.is_test = True if self.mode != propeller.RunMode.TRAIN else False
        self.run_config = run_config

    def forward(self, input_dict):
        gw = BatchGraphWrapper(input_dict['num_nodes'],
                               input_dict['num_edges'],
                               input_dict['edges'],
                               edge_feats={'efeat': input_dict['edge_feat']})

        feature = L.fc(input_dict['node_feat'], 
                    size=self.hparam.hidden_size,
                    act=None,
                    bias_attr=F.ParamAttr(name='embed_b'),
                    param_attr=F.ParamAttr(name="embed_w")
                    )

        edge_feature = L.fc(gw.edge_feat['efeat'], 
                    size=self.hparam.hidden_size,
                    act=None,
                    bias_attr=F.ParamAttr(name='edge_embed_b'),
                    param_attr=F.ParamAttr(name="edge_embed_w")
                    )

        for layer in range(self.hparam.num_layers):
            if layer == self.hparam.num_layers - 1:
                act = None
            else:
                act = 'leaky_relu'

            feature = GNNlayers.my_gnn(
                    gw,
                    feature,
                    edge_feature,
                    self.hparam.hidden_size,
                    act,
                    name="%s_%s" % (self.hparam.layer_type, layer))

        feature = L.dropout(
            feature,
            self.hparam.dropout_prob,
            dropout_implementation="upscale_in_train")

        logits = L.fc(feature, 
                size=self.hparam.num_class, 
                act=None,
                bias_attr=F.ParamAttr(name='final_b'),
                param_attr=F.ParamAttr(name="final_w"))

        mask = input_dict['mask']
        logits = paddle_helper.masked_select(logits, mask)

        return [logits, mask]

    def loss(self, predictions, label):
        logits = predictions[0]
        mask = predictions[1]
        label = paddle_helper.masked_select(label, mask)

        loss = L.mse_loss(input=logits, label=label)
        loss = L.reduce_mean(loss)

        return loss

    def backward(self, loss):
        optimizer = F.optimizer.Adam(learning_rate=self.hparam.lr)
        optimizer.minimize(loss)

    def metrics(self, predictions, label):
        result = {}
        logits = predictions[0]
        mask = predictions[1]
        label = paddle_helper.masked_select(label, mask)

        result["MCRMSE"] = propeller.metrics.MCRMSE(label, logits)

        return result


































































































