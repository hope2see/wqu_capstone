
import os
import time

import warnings
import numpy as np
import torch
import torch.nn as nn

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from models import TimesNet, DLinear, PatchTST, iTransformer, TimeMixer, TSMixer

from dataset_loader import get_data_provider

from basemodels import NeurlNetModel, StatisticalModel


# warnings.filterwarnings('ignore')

# NOTE
# Which one would be better? 
# 1) Basemodels are not trained after the first training (and validation) process.
# 2) Basemodels are trained continuously as a new input comes in.


from hyperopt import hp, tpe, rand, fmin, Trials, STATUS_OK
from sklearn.metrics import mean_absolute_error as MAE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from mem_util import MemUtil


class Metric:
    MAE = 0
    MSE = 1

class WeightingMethod:
    Inverted = 0
    SquaredInverted = 1
    Softmax = 2

class DefaultHP:
    lookback_window_size = 10
    metric = Metric.MAE
    weighting_method = WeightingMethod.Inverted
    discount_factor = 1.0  # no discounting
    smoothing_factor = 0.0  # no smoothing


def compute_comp_weights(hp_dict, models_loss, prev_comp_weights):
    lookback_window_size = int(hp_dict['lookback_window_size']) if 'lookback_window_size' in hp_dict else DefaultHP.lookback_window_size
    metric = hp_dict['metric'] if 'metric' in hp_dict else DefaultHP.metric
    weighting_method = hp_dict['weighting_method'] if 'weighting_method' in hp_dict else DefaultHP.weighting_method
    discount_factor = hp_dict['discount_factor'] if 'discount_factor' in hp_dict else DefaultHP.discount_factor
    smoothing_factor = hp_dict['smoothing_factor'] if 'smoothing_factor' in hp_dict else DefaultHP.smoothing_factor

    num_all_components = models_loss.shape[0]
    if 'max_components' in hp_dict: 
        max_components = min(int(hp_dict['max_components']), num_all_components)
    else:
        max_components = num_all_components

    # TODO ! 
    # cool_start = int(hp_dict['cool_start'])

    # comp_errors_window.shape : [num_component_models, lookback_window_size]
    comp_errors_window = models_loss[:, -lookback_window_size:]

    # apply discounting to error_matrix, and compute the component_error
    tau = lookback_window_size # NOTE : Do we need to separate tau and lookback_window_size?
    try:
        discount = np.power(discount_factor, np.arange(1,tau+1))
        discounted_comp_error_window = np.multiply(comp_errors_window, discount)
    except:
        assert False, "Error in discounting!"
        print("Error in discounting!")
        discounted_comp_error_window = comp_errors_window

    if metric == Metric.MSE:
        comp_errors = np.mean(np.power(discounted_comp_error_window,2), axis=1)
    else: # metric == Metric.MAE:
        comp_errors = np.mean(discounted_comp_error_window, axis=1)

    comp_errors = np.array([1e-5 if i<1e-12 else i for i in comp_errors])

    # compute the weights of component models by applying the weighting method
    if weighting_method == WeightingMethod.Softmax:
        if np.sum(np.exp(-comp_errors)) <1e-10:
            comp_errors = comp_errors / np.min(comp_errors)
        comp_weights = np.exp(-comp_errors) / np.sum(np.exp(-comp_errors))
    elif weighting_method == WeightingMethod.Inverted:
        comp_weights = np.power(comp_errors,-1) / np.sum(np.power(np.abs(comp_errors),-1))
    else: # weighting_method == WeightingMethod.SquaredInverted':
        comp_weights = np.power(comp_errors,-2) / np.sum(np.power(comp_errors,-2))

    # NOTE : Necesssary? 
    if np.isnan(comp_weights).any():
        # print("Nan in comp_weights")
        assert False, "Nan in comp_weights"
        comp_weights = np.nan_to_num(comp_weights) 
        comp_weights = comp_weights / np.sum(comp_weights)

    assert np.linalg.norm(np.sum(comp_weights) - 1.0) < 1e-3   

    # smoothing the weights
    if prev_comp_weights is not None:
        comp_weights = smoothing_factor * prev_comp_weights + (1-smoothing_factor) * comp_weights

    # apply max_components constraint : choose max_components components with the highest weights
    if max_components < num_all_components:
        chosen_indices = np.argpartition(comp_weights, -max_components)[-max_components:] 
        comp_weights = np.array([comp_weights[i] if i in chosen_indices else 0 for i in range(num_all_components)])
        comp_weights = comp_weights / np.sum(comp_weights)

    return comp_weights


# class for Hyper-Parameter Optimization
class HPO(object):
    
    def __init__(self, y, models_loss, use_BOA=True):
        self.y = y  # ground truth of forecast 
        self.models_loss = models_loss # losses of base models
        self.use_BOA = use_BOA

    def optimize_HP(self, hp_space, max_evals=100):
        num_comps = self.models_loss.shape[0]
        trials = Trials()
        algo = tpe.suggest if self.use_BOA else rand.suggest
        # algo = partial(algo, n_startup_jobs=1) 
        best_hp = fmin(self._evaluate_hp, hp_space, algo=algo, max_evals=max_evals, 
                       trials=trials, rstate=np.random.default_rng(1), verbose=True)
        return best_hp, trials


    # Objective function (loss function) for hyper-parameter optimization
    # Loss == Mean of the lossees in all timesteps in the period [lookback_window_size, len(y)]
    def _evaluate_hp(self, hp_dict):
        losses = []
        lookback_window_size = int(hp_dict['lookback_window_size'])
        comp_weights = None
        t = lookback_window_size
        while t < len(self.y):
            comp_weights = compute_comp_weights(
                hp_dict,
                self.models_loss[:, t-lookback_window_size : t], 
                comp_weights)
            next_y_hat = np.dot(comp_weights, self.models_loss[:, t:t+1])
            next_y = self.y[t]
            losses.append(np.abs(next_y_hat - next_y))
            t += 1

        mean_loss = np.mean(losses)
        var_loss = np.var(losses)

        return {
            # TODO : add comp_weights 
            'loss': mean_loss,         
            'loss_variance': var_loss, 
            'status': STATUS_OK
        }


def show_hpo_result(HP, trials, title):
    print("\nHyperparameters:")
    hp_df = pd.DataFrame(HP, index=[0])
    print(hp_df)

    # Plot the optimization progress (loss and loss variance) 
    losses = [t['result']['loss'] for t in trials]           
    variances = [t['result']['loss_variance'] for t in trials]

    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o', label='Mean Loss')
    plt.plot(variances, marker='x', label='Loss Variance')
    if title:
        plt.title(title)
    plt.xlabel("Trial Number")
    plt.ylabel("Value")
    plt.legend()    
    plt.grid(True)
    plt.show()


def plot_multiple_trials(trials_list, trial_labels, value_key, title=None):
    plt.figure(figsize=(8, 5))
    for i, trials in enumerate(trials_list):
        values = [t['result'][value_key] for t in trials] 
        plt.plot(values, marker='o', label=trial_labels[i])
    if title:
        plt.title(title)
    plt.xlabel("Trial Number")
    plt.ylabel(value_key)
    plt.legend()    
    plt.grid(True)
    plt.show()


def plot_forecast(y, y_hat, title=None):
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame({
        'Ground Truth': y,
        'Forecast': y_hat
    })
    # plt.plot(df)
    df.plot()
    if title:
        plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()    
    plt.grid(True)
    plt.show()


def plot_weights(weights_hist, title=None):
    plt.figure(figsize=(8, 5))
    for i in range(weights_hist.shape[1]):
        plt.plot(weights_hist[:,i], label=f'Component {i}')
    if title:
        plt.title(title)
    plt.ylim(0, 1)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()    
    plt.grid(True)
    plt.show()


class CombinerModel(object):
    max_lookback_window_size = 15

    def __init__(self, configs, basemodels, result_dir):
        self.configs = configs
        self.basemodels = basemodels
        self.result_dir = result_dir

        self.stat_models, self.nn_models = self.seperate_models()
        # Hyperparameter Space
        self.hp_space = {
            # 'cool_start': hp.quniform('cool_start', 0, num_comps-1, 1),
            'lookback_window_size': hp.quniform('lookback_window_size', 1, self.max_lookback_window_size, 1),
            'max_components': hp.quniform('max_components', 1, len(basemodels), 1),
            'metric': hp.choice('metric', [Metric.MAE, Metric.MSE]),
            'weighting_method': hp.choice('weighting_method', 
                                            [WeightingMethod.Inverted, WeightingMethod.SquaredInverted, WeightingMethod.Softmax]),
            'discount_factor':hp.uniform('discount_factor', 1.0, 1.5),
            'smoothing_factor': hp.uniform('smoothing_factor', 0.0, 1),
        }

        self.last_comp_weights = None


    def seperate_models(self):
        stat_models = []
        nn_models = []
        for model in self.basemodels:
            if isinstance(model, StatisticalModel): 
                stat_models.append(model)
            elif isinstance(model, NeurlNetModel): 
                nn_models.append(model)
            else: 
                assert False, "Model is not of type StatisticalModel or NeuralNetModel"
        return stat_models, nn_models

    # NOTE: TBD
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def train(self):
        # Train combiner model -------------------------------------
        time_now = time.time()

        train_dataset, train_loader = get_data_provider(self.configs, flag='ensemble_train', step_by_step=True)

        # prepare the forecasted values of base models         
        basemodel_preds = np.empty((len(self.basemodels), len(train_loader)))
        basemodel_losses = np.empty((len(self.basemodels), len(train_loader)))
        criterion = self._select_criterion()

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for m, basemodel in enumerate(self.basemodels):
                basemodel_losses[m, t], basemodel_preds[m, t] = basemodel.proceed_onestep(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, 
                    training=True) # NOTE : Base models ARE trained in ensemble_training period.
            MemUtil.print_memory_usage()

        spent_time = (time.time() - time_now) 
        print(f'CombinerModel.train() : {spent_time:.4f}sec elapsed for getting predition data from base models')

        # Hyperparameter Optimization -------------------------------------
        time_now = time.time()
        hp_boa, trials_boa = HPO(
            # data_y.shape = (seq_len + len_vali_data, features_dim). last column is the target feature
            y = train_dataset.data_y[self.configs.seq_len:, -1], 
            models_loss = basemodel_losses, 
            use_BOA = True
        ).optimize_HP(self.hp_space, max_evals=100)

        spent_time = (time.time() - time_now) 
        print(f'CombinerModel.train() : {spent_time:.4f}sec elapsed for hyperparameter optimization')

        show_hpo_result(hp_boa, trials_boa, "Bayesian Optimization for HPO")

        self.hp_dict = hp_boa
        # save the losses of base models for the test phase
        self.last_basemodel_losses = basemodel_losses[:, -self.max_lookback_window_size:] 
        

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training: bool = True):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        basemodel_preds = np.empty((len(self.basemodels)))
        basemodel_losses = np.empty((len(self.basemodels), 1))

        for m, basemodel in enumerate(self.basemodels):
            basemodel_losses[m, 0], basemodel_preds[m] = basemodel.proceed_onestep(
                batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training) 

        # concatenate the last vali losses to test losses. 
        basemodel_losses = np.concatenate((self.last_basemodel_losses, basemodel_losses), axis=1)
        bl_t = self.max_lookback_window_size

        lookback_window_size = int(self.hp_dict['lookback_window_size'])
        comp_weights = compute_comp_weights(
            self.hp_dict,
            basemodel_losses[:, bl_t-lookback_window_size:bl_t], 
            self.last_comp_weights)

        y_hat = np.dot(comp_weights, basemodel_preds)
        y = batch_y[0, -1, -1] 
        # loss = criterion(torch.tensor(y_hat), y).item()

        self.last_comp_weights = comp_weights
        self.last_basemodel_losses = basemodel_losses[:, -self.max_lookback_window_size:] 
        return y_hat, basemodel_preds


    def test(self):
        time_now = time.time()

        test_data, test_loader = get_data_provider(self.configs, flag='test', step_by_step=True)
        y = test_data.data_y[self.configs.seq_len:, -1]
        need_to_invert_data = True if (test_data.scale and self.configs.inverse) else False

        # prepare the forecasted values of base models in test period
        basemodel_preds = np.empty((len(self.basemodels), len(test_loader)))
        basemodel_losses = np.empty((len(self.basemodels), len(test_loader)))
        criterion = self._select_criterion()
        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for m, basemodel in enumerate(self.basemodels):
                basemodel_preds[m, t] = basemodel.proceed_onestep(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training=True)      
                
        # compute CombinerModel's predictions
        y_hat = np.empty_like(y)
        weights_hist = np.empty((len(y), len(self.basemodels)))

        # concatenate the last vali losses to test losses. 
        basemodel_losses = np.concatenate((self.last_basemodel_losses, basemodel_losses), axis=1)
        bl_t = self.max_lookback_window_size

        lookback_window_size = int(self.hp_dict['lookback_window_size'])
        comp_weights = None
        for t in range(len(y)):
            comp_weights = compute_comp_weights(
                self.hp_dict,
                basemodel_losses[:, bl_t-lookback_window_size:bl_t], 
                comp_weights)
            y_hat[t] = np.dot(comp_weights, basemodel_preds[:, t:t+1])
            weights_hist[t] = comp_weights
            bl_t += 1

        spent_time = (time.time() - time_now) 
        print(f'CombinerModel.test() : {spent_time:.4f}sec elapsed for testing')

        plot_weights(weights_hist, title="Component Weights")

        if need_to_invert_data:
            n_features = test_data.data_y.shape[1]
            data_y = np.zeros((len(y), n_features))
            data_y_hat = np.zeros((len(y), n_features))
            data_y[:, -1] = y
            data_y_hat[:, -1] = y_hat
            y = test_data.inverse_transform(data_y)[:, -1]
            y_hat = test_data.inverse_transform(data_y_hat)[:, -1]

        losses = criterion(torch.tensor(y_hat), y)

        print(f"CombinerModel.test() : Loss ----- ")
        print(f"max={np.max(losses):.6f}, mean={np.mean(losses):.6f}, min={np.min(losses):.6f}, var={np.var(losses):.6f})")

        # plot_forecast(y, y_hat)

        return y, y_hat, losses

