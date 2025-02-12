
# ".env" file is used instead of the sys.path.append() method
# import sys
# Add Time-Series-Library directory to module lookup paths
# sys.path.append('Time-Series-Library')

import os 
import argparse
from utils.print_args import print_args
# from data_provider import data_factory, data_loader
from models import TimesNet, DLinear, PatchTST, iTransformer, TimeXer, TSMixer
# from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics import MAE, MSE, RMSE, MAPE, MSPE

import matplotlib.pyplot as plt

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim

from basemodels import EtsModel, SarimaModel, NeurlNetModel
from combiner import CombinerModel
from adjuster import AdjusterModel

def get_setting_str(args):
    setting_str = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des)
    return setting_str


def parse_cmd_args(args=None):
    # parser = argparse.ArgumentParser(description='TimesNet')
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

    args = parser.parse_args(args)
    return args


def set_device_configs(args):
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


def plot_forecast_result(truth, adjuster_pred, combiner_pred, base_preds, basemodels, filepath='./result_forecast_result.pdf'):
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


metric_dict = {
    "MAE": MAE, 
    "MSE": MSE, 
    "RMSE": RMSE, 
    "MAPE": MAPE, 
    "MSPE": MSPE
}


def report_losses(y, y_hat_adj, y_hat_cbm, y_hat_bsm, filepath=None):
    losses_adj = {}
    losses_cbm = {}
    losses_bsm = {}

    if filepath is not None:
        f = open(filepath, 'a')

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
                    +  ", ".join([f"{loss:.4f}" for loss in losses_bsm[m]])
        print(output_str)
        if filepath is not None:
            f.write(output_str + '\n')

    if filepath is not None:
        f.close()

    return losses_adj, losses_cbm, losses_bsm


def cleanup_gpu_cache(args):
    if args.gpu_type == 'mps':
        # Only if mps.empty_cache() is available, then call it
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()


def run(args=None):
    configs = parse_cmd_args(args)
    set_device_configs(configs)

    print('\nConfigurations =================================')
    print_args(configs)

    # fix seed of random funcs
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # create base models 
    etsModel = EtsModel(configs)
    sarimaModel = SarimaModel(configs)
    dLinearModel = NeurlNetModel(configs, "DLinear", DLinear)
    iTransformerModel = NeurlNetModel(configs, "iTransformer", iTransformer)
    timeXerModel = NeurlNetModel(configs, "TimeXer", TimeXer)
    # basemodels = [etsModel, sarimaModel, dLinearModel]
    basemodels = [etsModel, sarimaModel, dLinearModel, iTransformerModel, timeXerModel]
    # basemodels = [etsModel, sarimaModel]

    setting_str = get_setting_str(configs)

    # create combiner model
    combinerModel = CombinerModel(configs, basemodels, setting_str)

    # create adjuster model 
    adjusterModel = AdjusterModel(configs, combinerModel)

    # train base models 
    if configs.is_training:
        print('\nTraining base models ==================================')
        for basemodel in basemodels:
            # TODO : Better to optimize the the reduncancy of the same proceudures in training base models. 
            print(f'\nTraining {basemodel.name} ...')
            basemodel.train(setting_str)
    else:
        print('\nLoading trained base models ======================')
        for basemodel in basemodels:
            basemodel.load_saved_model(setting_str)

    print('\nTraining combiner model ======================')
    combinerModel.train()

    print('\nTraining adjuster model ======================')
    adjusterModel.train()

    print('\nTesting ==================================')
    y, y_hat, losses, y_hat_cbm, losses_cbm, y_hat_bsm = adjusterModel.test()

    report_losses(y, y_hat, y_hat_cbm, y_hat_bsm, filepath="./result_losses.txt")

    # Plot forecast results of all models
    plot_forecast_result(y, y_hat, y_hat_cbm, y_hat_bsm, basemodels)

    # Plot (and save to file) the result of the last lookback window of test period 
    adjusterModel.plot_gpmodel()

    # clean-up cache 
    cleanup_gpu_cache(configs)

    print('Bye ~~~~~~')


if __name__ == '__main__':
    run()
