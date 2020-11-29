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

import pgl
from pgl.utils import paddle_helper
from pgl.utils.data.dataloader import Dataloader
from propeller import log
log.setLevel(logging.DEBUG)
import propeller.paddle as propeller
from propeller.paddle.data import Dataset as PDataset

import paddle.fluid as F
import paddle.fluid.layers as L

from utils.config import prepare_config, make_dir
from utils.logger import prepare_logger, log_to_file
from utils.util import int82strarr
from dataset import random_split, CovidDataset, CollateFn
from model import GNNModel

def multi_epoch_dataloader(loader, epochs):
    def _worker():
        for i in range(epochs):
            log.info("BEGIN: epoch %s ..." % i)
            for batch in loader():
                yield batch
            log.info("END: epoch %s ..." % i)
    return _worker

def build_id_seqpos(data_file):
    df = pd.read_json(data_file, lines=True)
    id_seqpos = []
    for i in range(len(df)):
        id = df.loc[i, 'id']
        seq_length = df.loc[i, 'seq_length']
        for seqpos in range(seq_length):
            id_seqpos.append(id + '_' + str(seqpos))
    return id_seqpos

def train(args):
    train_ds = CovidDataset(data_file=args.train_file, config=args, mode="train")
    valid_ds = CovidDataset(data_file=args.valid_file, config=args, mode="valid")

    log.info("train examples: %s" % len(train_ds))
    log.info("valid examples: %s" % len(train_ds))

    train_loader = Dataloader(train_ds, 
                              batch_size=args.batch_size,
                              shuffle=args.shuffle,
                              collate_fn=CollateFn())
    train_loader = multi_epoch_dataloader(train_loader, args.epochs)
    train_loader = PDataset.from_generator_func(train_loader)

    valid_loader = Dataloader(valid_ds, 
                            batch_size=1,
                            shuffle=False,
                            collate_fn=CollateFn())
    valid_loader = PDataset.from_generator_func(valid_loader)

    # warmup start setting
    ws = None
    propeller.train.train_and_eval(
            model_class_or_model_fn=GNNModel,
            params=args,
            run_config=args,
            train_dataset=train_loader,
            eval_dataset={"eval": valid_loader},
            warm_start_setting=ws,
            )


def infer(args):
    # predict for test data
    log.info('Reading %s' % args.test_file)
    test_ds = CovidDataset(args.test_file, args, 'test')
    test_loader = Dataloader(test_ds,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn=CollateFn(mode='test'))
    test_loader = PDataset.from_generator_func(test_loader)

    est = propeller.Learner(GNNModel, args, args)

    output_path = args.model_path_for_infer.replace("checkpoints/", "outputs/")
    make_dir(output_path)
    filename = os.path.join(output_path, "submission.csv")

    id_seqpos = build_id_seqpos(args.test_file)

    preds = []
    for predictions in est.predict(test_loader,
                                   ckpt_path=args.model_path_for_infer, 
                                   split_batch=False):
        preds.append(predictions[0])

    preds = np.concatenate(preds)
    df_sub = pd.DataFrame({'id_seqpos': id_seqpos,
                           'reactivity': preds[:,0],
                           'deg_Mg_pH10': preds[:,1],
                           'deg_pH10': 0,
                           'deg_Mg_50C': preds[:,2],
                           'deg_50C': 0})
    log.info("saving result to %s" % filename)
    df_sub.to_csv(filename, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    if args.mode == "infer":
        config = prepare_config(args.config, isCreate=False, isSave=False)
        infer(config)
    else:
        config = prepare_config(args.config, isCreate=True, isSave=True)
        log_to_file(log, config.log_dir)
        train(config)































































































