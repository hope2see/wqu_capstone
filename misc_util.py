
# import os 
# import argparse
import numpy as np
import torch


def get_config_str(configs):
    setting_str = f"{configs.model_id}_{configs.model_id}_{configs.data}_sl{configs.seq_len}"
    setting_str += f"_ahpo_{configs.hpo_interval}" if configs.adaptive_hpo else "_ahpo_no"
    setting_str = f"_ep{configs.train_epochs}_"
    setting_str += configs.des
    return setting_str


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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

    # In order to do 'continous train()', we should keep tracking best_model. 
    # We use one EarlyStopping instance to track best_model while resetting it when needed. 
    # So, self.best_score should not be reset
    def reset(self):
        self.early_stop = False
        self.counter = 0
