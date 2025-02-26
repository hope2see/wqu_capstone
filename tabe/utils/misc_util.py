
import os 
import logging
import numpy as np
import torch
import sys
import typing
import time 
from datetime import datetime

_datatime_sig = datetime.now().strftime("%Y_%m_%d_%H%M%S")

# logging -----------------------------------------------------

_DEFAULT_LOGGER = "tabe_logger"

_DEFAULT_FORMATTER = logging.Formatter(
    '%(asctime)s - %(filename)s[%(funcName)s] - %(levelname)s: %(message)s'
)
# _DEFAULT_FORMATTER = logging.Formatter(
#     '%(asctime)s - %(filename)s[pid:%(process)d;line:%(lineno)d:%(funcName)s] - %(levelname)s: %(message)s'
# )

_h_stdout = logging.StreamHandler(stream=sys.stdout)
_h_stdout.setFormatter(_DEFAULT_FORMATTER)
_h_stdout.setLevel(logging.DEBUG)

_h_file = logging.FileHandler(f'tabe_log_{_datatime_sig}.log', mode='w')  
_h_file.setFormatter(_DEFAULT_FORMATTER)
_h_file.setLevel(logging.INFO)

_DEFAULT_HANDLERS = [_h_stdout, _h_file]

_LOGGER_CACHE = {}  # type: typing.Dict[str, logging.Logger]

def get_logger(name, level="DEBUG", handlers=None, update=False, verbose=False):
    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers or _DEFAULT_HANDLERS
    logger.propagate = False
    return logger

logger = get_logger(_DEFAULT_LOGGER)


#  -----------------------------------------------------

def get_config_str(configs):
    setting_str =  f"{configs.des}_{configs.model_id}_sl{configs.seq_len}"
    setting_str += f"_ahpo_{configs.hpo_interval}" if configs.adaptive_hpo else "_ahpo_no"
    setting_str += f"_ep{configs.train_epochs}_" + configs.data_path + "_"
    setting_str += _datatime_sig
    return setting_str


#  -----------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # np.Inf` was removed in the NumPy 2.0 release. Use `np.inf` instead.
        self.val_loss_min = np.inf # np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

    # In order to do 'continous train()', we should keep tracking best_model. 
    # We use one EarlyStopping instance to track best_model while resetting it when needed. 
    # So, self.best_score should not be reset
    def reset(self):
        self.early_stop = False
        self.counter = 0

#  -----------------------------------------------------


# def simulate_trading(pred):
