# ".env" file is used instead of the sys.path.append() method
# import sys
# Add Time-Series-Library directory to module lookup paths
# sys.path.append('Time-Series-Library')

import os 
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim

# Time-Series-Library
from utils.print_args import print_args
from models import TimesNet, DLinear, PatchTST, iTransformer, TimeXer, TSMixer
from utils.metrics import MAE, MSE, RMSE, MAPE, MSPE

# TABE 
from tabe.abstractmodel import AbstractModel
from tabe.basemodels import EtsModel, SarimaModel, TSLibModel
from tabe.timemoe import TimeMoE
from tabe.combiner import CombinerModel
from tabe.adjuster import AdjusterModel
from utils.misc_util import get_config_str
from utils.mem_util import MemUtil

_mem_util = MemUtil(rss_mem=True, python_mem=True)


def _set_seed(fix_seed = 2025):
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


def _comma_separated_list(arg_value):
    return [v.strip() for v in arg_value.split(',')]


def _parse_cmd_args(args=None):
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # CMamba
    parser.add_argument('--dt_rank', type=int, default=32)
    parser.add_argument('--patch_num', type=int, default=32)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--dt_min', type=float, default=0.001)
    parser.add_argument('--dt_init', type=str, default='random', help='random or constant')
    parser.add_argument('--dt_max', type=float, default=0.1)
    parser.add_argument('--dt_scale', type=float, default=1.0)
    parser.add_argument('--dt_init_floor', type=float, default=1e-4)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--conv_bias', type=bool, default=True)
    parser.add_argument('--pscan', action='store_true', help='use parallel scan mode or sequential mode when training', default=False)
    parser.add_argument('--avg', action='store_true', help='avg pooling', default=False)
    parser.add_argument('--max', action='store_true', help='max pooling', default=False)
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--gddmlp', action='store_true', help='global data-dependent mlp', default=False)
    parser.add_argument('--channel_mixup', action='store_true', help='channel mixup', default=False)
    parser.add_argument('--sigma', type=float, default=1.0)

    # Combiner
    parser.add_argument('--basemodels', type=_comma_separated_list, default=[], 
                        help="Comma-separated names of base models to be inlcuded in the combiner model")
    parser.add_argument('--adaptive_hpo', default=False, action="store_true", help="apply Adaptive HPO in combiner model")
    parser.add_argument('--hpo_interval', type=int, default=1, help="interval (timesteps >= 1) for Adaptive HPO")

    args = parser.parse_args(args)
    return args


def _set_device_configs(args):
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        args.gpu_type = 'cuda' # by default
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            args.device = torch.device("mps")
            args.gpu_type = 'mps' # If not set, it causes an error in Exp_Basic._acquire_device()
            print('Using mps')
        else:
            args.device = torch.device("cpu")
            print('Using cpu')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


def _create_base_model(configs, model_name) -> AbstractModel:
    tslib_models = { # models in Time-Series-Library
        # 'TimesNet': TimesNet,
        'DLinear': DLinear,
        'PatchTST': PatchTST,
        'iTransformer': iTransformer,
        'TimeXer': TimeXer,
        # 'TSMixer': TSMixer
    }
    other_models = {
        'EtsModel': EtsModel,
        'SarimaModel': SarimaModel,
        'TimeMoE': TimeMoE,
    }

    if model_name in tslib_models:
        model = TSLibModel(configs, model_name, tslib_models[model_name])
    elif model_name in other_models:
        model = other_models[model_name](configs)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model


def plot_forecast_result(truth, adjuster_pred, combiner_pred, base_preds, basemodels, filepath):
    plt.figure(figsize=(12, 6))
    plt.title('Forecast Comparison')     
    plt.ylabel('Target (BTC return in 25 days)')     
    plt.xlabel('Test Duration (Days)')
    plt.plot(truth, label='GroundTruth', linewidth=2, color='black')
    plt.plot(adjuster_pred, label="Adjuster Model", linewidth=2, color='red')
    plt.plot(combiner_pred, label="Combiner Model", linewidth=2, color='blue')
    for i, basemodel in enumerate(basemodels):
        plt.plot(base_preds[i], label=f"Base Model [{basemodel.name}]", linewidth=1)
    plt.legend()
    plt.savefig(filepath, bbox_inches='tight')


def _report_losses(y, y_hat_adj, y_hat_cbm, y_hat_bsm, filepath=None):
    metric_dict = {
        "MAE": MAE, 
        "MSE": MSE, 
        "RMSE": RMSE, 
        "MAPE": MAPE, 
        "MSPE": MSPE
    }

    losses_adj = {}
    losses_cbm = {}
    losses_bsm = {}

    if filepath is not None:
        f = open(filepath, 'w')

    print("\n--------------------------------")
    print("Losses of all models")
    print("--------------------------------")

    for m in metric_dict:
        losses_adj[m] = metric_dict[m](y_hat_adj, y)
        losses_cbm[m] = metric_dict[m](y_hat_cbm, y)
        losses_bsm[m] = []
        for i in range(len(y_hat_bsm)):
            losses_bsm[m].append(metric_dict[m](y_hat_bsm[i], y))
        output_str = f"{m:<4}: {losses_adj[m]:.6f}, {losses_cbm[m]:.6f}, " \
                    +  ", ".join([f"{loss:.6f}" for loss in losses_bsm[m]])
        print(output_str)
        if filepath is not None:
            f.write(output_str + '\n')

    if filepath is not None:
        f.close()

    return losses_adj, losses_cbm, losses_bsm


def _cleanup_gpu_cache(args):
    if args.gpu_type == 'mps':
        # Only if mps.empty_cache() is available, then call it
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()


def run(args=None):
    _set_seed()

    _mem_util.start_python_memory_tracking()
    _mem_util.print_memory_usage()

    configs = _parse_cmd_args(args)
    _set_device_configs(configs)
    print_args(configs)

    basemodels = []
    for model_name in configs.basemodels:
        basemodels.append(_create_base_model(configs, model_name))
    
    combinerModel = CombinerModel(configs, basemodels)
    adjusterModel = AdjusterModel(configs, combinerModel)

    if configs.is_training:
        print('\nTraining base models ==================================')
        for basemodel in basemodels:
            # TODO : Better to optimize the the reduncancy of the same proceudures in training base models. 
            print(f'\nTraining {basemodel.name} ...')
            basemodel.train()
    else:
        print('\nLoading trained base models ======================')
        for basemodel in basemodels:
            basemodel.load_saved_model()

    _mem_util.print_memory_usage()

    print('\nTraining combiner model ======================')
    combinerModel.train()
    _mem_util.print_memory_usage()

    print('\nTraining adjuster model ======================')
    adjusterModel.train()
    _mem_util.print_memory_usage()

    print('\nTesting ==================================')
    y, y_hat, y_hat_cbm, y_hat_bsm = adjusterModel.test()
    _mem_util.print_memory_usage()

    result_dir = "./result/" + get_config_str(configs)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    _report_losses(y, y_hat, y_hat_cbm, y_hat_bsm, 
                  filepath = result_dir + "/models_losses.txt")

    plot_forecast_result(y, y_hat, y_hat_cbm, y_hat_bsm, basemodels, 
                        filepath = result_dir + "/models_forecast_comparison.pdf")

    adjusterModel.plot_gpmodel()

    _cleanup_gpu_cache(configs)

    _mem_util.stop_python_memory_tracking()

    print('Bye ~~~~~~')


if __name__ == '__main__':
    run()
