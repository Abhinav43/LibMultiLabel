import copy
import json
import logging
import os
import time
import pandas as pd

import numpy as np
import torch
from pytorch_lightning.utilities.seed import seed_everything


class Timer(object):
    """Computes elasped time."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


def dump_log(log_path, metrics=None, split=None, config=None):
    """Write log including config and the evaluation scores.
    Args:
        log_path(str): path to log path
        metrics (dict): metric and scores in dictionary format, defaults to None
        split (str): val or test, defaults to None
        config (dict): config to save, defaults to None
    """
    
    
    confg_result = log_path
    log_path.replace('logs.json','result.csv')
    confg_result.replace('logs.json','config_result.txt')
#     log_path_new = log_path.split('/')[2]
    
    
    if config:
        config_to_save = copy.deepcopy(dict(config))
        with open(confg_result,'a') as f:
            f.write(str(config_to_save) + '\n')
    
    if split and metrics:
        if os.path.isfile(log_path):
            df     = pd.read_csv(log_path)
        else:
            df     = pd.DataFrame({'AUC-ROC macro': [0], 'AUC-ROC micro': [0], 
                      'Another-Macro-F1': [0],'Macro-F1': [0],
                      'Micro-F1': [0],'P@5': [0]})
        
        metrics = {k: [m] for k,m in metrics.items()}
        result = pd.DataFrame(metrics)
        df_new = pd.concat([df, result])
        df_new.to_csv(log_path, index=False)
        
    logging.info(f'Finish writing log to {log_path}.')


def set_seed(seed):
    """Set seeds for numpy and pytorch."""
    if seed is not None:
        if seed >= 0:
            seed_everything(seed=seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        else:
            logging.warning(
                f'the random seed should be a non-negative integer')


def init_device(use_cpu=False):
    if not use_cpu and torch.cuda.is_available():
        # Set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
        # https://docs.nvidia.com/cuda/cublas/index.html
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy('file_system')
    logging.info(f'Using device: {device}')
    return device


def argsort_top_k(vals, k, axis=-1):
    unsorted_top_k_idx = np.argpartition(vals, -k, axis=axis)[:,-k:]
    unsorted_top_k_scores = np.take_along_axis(vals, unsorted_top_k_idx, axis=axis)
    sorted_order = np.argsort(-unsorted_top_k_scores, axis=axis)
    sorted_top_k_idx = np.take_along_axis(unsorted_top_k_idx, sorted_order, axis=axis)
    return sorted_top_k_idx
