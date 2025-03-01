
import numpy as np
import torch
import time 
from datetime import datetime
from tabe.utils.logger import logger
from tabe.utils.losses import mape_loss, smape_loss, mase_loss
import torch.nn as nn


_experiment_signature = None
def set_experiment_sig(configs):
    global _experiment_signature
    _experiment_signature =  f"{configs.model_id}_sl{configs.seq_len}"
    _experiment_signature += f"_ep{configs.train_epochs}_" 
    _experiment_signature += datetime.now().strftime("%Y%m%d_%H%M%S")

def experiment_sig():
    assert _experiment_signature is not None
    return _experiment_signature

#  -----------------------------------------------------

def get_loss_func(loss:str):
    loss = loss.upper()
    if loss == 'MSE':
        return nn.MSELoss()
    elif loss == 'MAE':
        return nn.L1Loss()
    elif loss == 'MAPE':
        return mape_loss
    elif loss == 'SMAPE':
        return smape_loss
    elif loss == 'MASE':
        return mase_loss
    else:
        logger.error('Not suppprted loss : {loss}')
        raise NotImplementedError

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



# -----------

def print_configs(configs):
    logger.info(f"\033[1m" + "TABE Config" + "\033[0m")
    logger.info(f'  {"Model ID:":<20}{configs.model_id:<20}{"Model:":<20}{configs.model:<20}')
    logger.info(f'  {"Task Name:":<20}{configs.task_name:<20}{"Is Training:":<20}{configs.is_training:<20}')
    logger.info('')

    logger.info(f"\033[1m" + "Adjuster Configs" + "\033[0m")
    if configs.adjuster is not None:
        logger.info(f'  {configs.adjuster}')
    logger.info('')

    logger.info(f"\033[1m" + "Combiner Configs" + "\033[0m")
    if configs.combiner is not None:
        logger.info(f'  {configs.combiner}')
    logger.info('')

    # logger.info(f"\033[1m" + "Adaptive HPO : {configs.adaptive_hpo}" + "\033[0m")
    # if configs.adaptive_hpo:
    #     logger.info(f'  {"interval:":<20}{configs.hpo_interval:<20}{"max_hpo_eval:":<20}{configs.label_len:<20}')
    # logger.info()

    logger.info(f"\033[1m" + "Base Models" + "\033[0m")
    for (model_name, model_args) in configs.basemodel:
        logger.info(f'   {model_name:<10}{model_args} ')
    logger.info('')

    logger.info("\033[1m" + f"Data {configs.data:<20}" + "\033[0m")
    logger.info(f'  {"Features:":<20}{configs.features:<20}{configs.target:<20}{"Freq:":<20}{configs.freq:<20}')
    logger.info(f'  {"Target datatype:":<20}{configs.target_datatype:<20}{"Target Column:":<20}')
    logger.info(f'  {"Root Path:":<20}{configs.root_path:<20}{"Data Path:":<20}{configs.data_path:<20}')
    logger.info(f'  {"Checkpoints:":<20}{configs.checkpoints:<20}')
    logger.info('')

    if configs.task_name in ['long_term_forecast', 'short_term_forecast']:
        logger.info("\033[1m" + "Forecasting Task" + "\033[0m")
        logger.info(f'  {"Seq Len:":<20}{configs.seq_len:<20}{"Label Len:":<20}{configs.label_len:<20}')
        logger.info(f'  {"Pred Len:":<20}{configs.pred_len:<20}{"Seasonal Patterns:":<20}{configs.seasonal_patterns:<20}')
        logger.info(f'  {"Inverse:":<20}{configs.inverse:<20}')
        logger.info('')

    # if configs.task_name == 'imputation':
    #     logger.info("\033[1m" + "Imputation Task" + "\033[0m")
    #     logger.info(f'  {"Mask Rate:":<20}{configs.mask_rate:<20}')
    #     logger.info()

    # if configs.task_name == 'anomaly_detection':
    #     logger.info("\033[1m" + "Anomaly Detection Task" + "\033[0m")
    #     logger.info(f'  {"Anomaly Ratio:":<20}{configs.anomaly_ratio:<20}')
    #     logger.info()


    logger.info("\033[1m" + "Model Parameters" + "\033[0m")
    logger.info(f'  {"Top k:":<20}{configs.top_k:<20}{"Num Kernels:":<20}{configs.num_kernels:<20}')
    logger.info(f'  {"Enc In:":<20}{configs.enc_in:<20}{"Dec In:":<20}{configs.dec_in:<20}')
    logger.info(f'  {"C Out:":<20}{configs.c_out:<20}{"d model:":<20}{configs.d_model:<20}')
    logger.info(f'  {"n heads:":<20}{configs.n_heads:<20}{"e layers:":<20}{configs.e_layers:<20}')
    logger.info(f'  {"d layers:":<20}{configs.d_layers:<20}{"d FF:":<20}{configs.d_ff:<20}')
    logger.info(f'  {"Moving Avg:":<20}{configs.moving_avg:<20}{"Factor:":<20}{configs.factor:<20}')
    logger.info(f'  {"Distil:":<20}{configs.distil:<20}{"Dropout:":<20}{configs.dropout:<20}')
    logger.info(f'  {"Embed:":<20}{configs.embed:<20}{"Activation:":<20}{configs.activation:<20}')
    logger.info('')

    logger.info("\033[1m" + "Run Parameters" + "\033[0m")
    logger.info(f'  {"Num Workers:":<20}{configs.num_workers:<20}{"Itr:":<20}{configs.itr:<20}')
    logger.info(f'  {"Train Epochs:":<20}{configs.train_epochs:<20}{"Batch Size:":<20}{configs.batch_size:<20}')
    logger.info(f'  {"Patience:":<20}{configs.patience:<20}{"Learning Rate:":<20}{configs.learning_rate:<20}')
    logger.info(f'  {"Des:":<20}{configs.des:<20}{"Loss:":<20}{configs.loss:<20}')
    logger.info(f'  {"Lradj:":<20}{configs.lradj:<20}{"Use Amp:":<20}{configs.use_amp:<20}')
    logger.info('')

    logger.info("\033[1m" + "GPU" + "\033[0m")
    logger.info(f'  {"Use GPU:":<20}{configs.use_gpu:<20}{"GPU:":<20}{configs.gpu:<20}')
    logger.info(f'  {"Use Multi GPU:":<20}{configs.use_multi_gpu:<20}{"Devices:":<20}{configs.devices:<20}')
    logger.info('')

    # logger.info("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    # p_hidden_dims_str = ', '.join(map(str, configs.p_hidden_dims))
    # logger.info(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{configs.p_hidden_layers:<20}') 
    # logger.info()
