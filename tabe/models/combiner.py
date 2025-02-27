# NOTE
# Which one would be better? 
# 1) Basemodels are not trained once it they have been trained in the first training (and validation) process.
# 2) Basemodels are trained continuously as a new input comes in.


import time
import numpy as np
import warnings
import numpy as np
import torch

from hyperopt import hp, tpe, rand, fmin, Trials, STATUS_OK
from sklearn.metrics import mean_absolute_error as MAE


from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from models import TimesNet, DLinear, PatchTST, iTransformer, TimeMixer, TSMixer

from tabe.data_provider.dataset_loader import get_data_provider
from tabe.models.abstractmodel import AbstractModel
import tabe.utils.report as report
from tabe.utils.mem_util import MemUtil
from tabe.utils.logger import logger


warnings.filterwarnings('ignore')


_mem_util = MemUtil(rss_mem=False, python_mem=False)


class _Metric:
    MAE = 0
    MSE = 1

class _WeightingMethod:
    Inverted = 0
    SquaredInverted = 1
    Softmax = 2

class _DefaultHP:
    lookback_window_size = 10
    metric = _Metric.MAE
    weighting_method = _WeightingMethod.Inverted
    discount_factor = 1.0  # no discounting
    smoothing_factor = 0.0  # no smoothing


class CombinerModel(AbstractModel):

    # Maximum lookback-window size for computing weights of base models
    MAX_LOOKBACK_WIN = 15 

    # Period for HPO (HyperParameter Optimization)
    # The most recent period of this length is used for HPO. 
    # For the first HPO, the ensemble-training period can be used. 
    # But, to provide continuous adaptive HPO feature, 
    # the latest input should be used (or added) for HPO. 
    # And, ever-growing input size is not practically acceptable. 
    # Thus, we use fixed-size evaluation period for HPO.
    MAX_HPO_EVAL_PEROID = MAX_LOOKBACK_WIN * 2 


    def __init__(self, configs, basemodels):
        super().__init__(configs, "Combiner")
        self.basemodels = basemodels
        self.hp_space = {
            # 'cool_start': hp.quniform('cool_start', 0, num_comps-1, 1),
            'lookback_window_size': hp.quniform('lookback_window_size', 1, self.MAX_LOOKBACK_WIN, 1),
            'max_components': hp.quniform('max_components', 1, len(basemodels), 1),
            'metric': hp.choice('metric', [_Metric.MAE, _Metric.MSE]),
            'weighting_method': hp.choice('weighting_method', 
                                            [_WeightingMethod.Inverted, _WeightingMethod.SquaredInverted, _WeightingMethod.Softmax]),
            'discount_factor':hp.uniform('discount_factor', 1.0, 1.5),
            'smoothing_factor': hp.uniform('smoothing_factor', 0.0, 1),
        }
        self.hp_dict = None # currently active hyper-parameters
        self.hpo_counter = 0 # used for counting timesteps for Adaptive HPO
        self.basemodel_weights = None # lastly applied weights for basemodels
        self.basemodel_losses = None # the last basemodel_losses. Shape = (len(basemodels), HPO_EVALUATION_PEROID)
        self.truths = None # the last 'y' values. Shape = (HPO_EVALUATION_PEROID)


    def compute_basemodel_weights(self, hp_dict, models_loss, prev_comp_weights):
        lookback_window_size = int(hp_dict['lookback_window_size']) if 'lookback_window_size' in hp_dict else _DefaultHP.lookback_window_size
        metric = hp_dict['metric'] if 'metric' in hp_dict else _DefaultHP.metric
        weighting_method = hp_dict['weighting_method'] if 'weighting_method' in hp_dict else _DefaultHP.weighting_method
        discount_factor = hp_dict['discount_factor'] if 'discount_factor' in hp_dict else _DefaultHP.discount_factor
        smoothing_factor = hp_dict['smoothing_factor'] if 'smoothing_factor' in hp_dict else _DefaultHP.smoothing_factor

        num_all_components = models_loss.shape[0]
        if 'max_components' in hp_dict: 
            max_components = min(int(hp_dict['max_components']), num_all_components)
        else:
            max_components = num_all_components

        # TODO ! 
        # cool_start = int(hp_dict['cool_start'])

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

        if metric == _Metric.MSE:
            comp_errors = np.mean(np.power(discounted_comp_error_window,2), axis=1)
        else: # metric == Metric.MAE:
            comp_errors = np.mean(discounted_comp_error_window, axis=1)

        comp_errors = np.array([1e-5 if i<1e-12 else i for i in comp_errors])

        # compute the weights of component models by applying the weighting method
        if weighting_method == _WeightingMethod.Softmax:
            if np.sum(np.exp(-comp_errors)) <1e-10:
                comp_errors = comp_errors / np.min(comp_errors)
            basemodel_weights = np.exp(-comp_errors) / np.sum(np.exp(-comp_errors))
        elif weighting_method == _WeightingMethod.Inverted:
            basemodel_weights = np.power(comp_errors,-1) / np.sum(np.power(np.abs(comp_errors),-1))
        else: # weighting_method == WeightingMethod.SquaredInverted':
            basemodel_weights = np.power(comp_errors,-2) / np.sum(np.power(comp_errors,-2))

        # NOTE : Necesssary? 
        if np.isnan(basemodel_weights).any():
            # print("Nan in basemodel_weights")
            assert False, "Nan in basemodel_weights"
            basemodel_weights = np.nan_to_num(basemodel_weights) / np.sum(basemodel_weights)

        assert np.linalg.norm(np.sum(basemodel_weights) - 1.0) < 1e-3   

        # smoothing the weights
        if prev_comp_weights is not None:
            basemodel_weights = smoothing_factor * prev_comp_weights + (1-smoothing_factor) * basemodel_weights

        # apply max_components constraint : choose max_components components with the highest weights
        if max_components < num_all_components:
            chosen_indices = np.argpartition(basemodel_weights, -max_components)[-max_components:] 
            basemodel_weights = np.array([basemodel_weights[i] if i in chosen_indices else 0 for i in range(num_all_components)])
            basemodel_weights = basemodel_weights / np.sum(basemodel_weights)

        return basemodel_weights


    def _optimize_HP(self, use_BOA=True, max_evals=100):

        # Objective function (loss function) for hyper-parameter optimization
        # Loss == Mean of the lossees in all timesteps in the period [lookback_window_size, len(y)]
        def _evaluate_hp(hp_dict):
            lookback_window_size = int(hp_dict['lookback_window_size'])
            losses = []
            basemodel_weights = None
            t = self.MAX_LOOKBACK_WIN # We should start from the same timestep to evaluate the same period.
            while t < len(self.truths):
                basemodel_weights = self.compute_basemodel_weights(
                    hp_dict, self.basemodel_losses[:, t-lookback_window_size : t], basemodel_weights)
                next_y_hat = np.dot(basemodel_weights, self.basemodel_losses[:, t:t+1])
                next_y = self.truths[t]
                losses.append(np.abs(next_y_hat - next_y))
                t += 1
            mean_loss = np.mean(losses)
            var_loss = np.var(losses)

            return {
                # TODO : add basemodel_weights 
                'loss': mean_loss,         
                'loss_variance': var_loss, 
                'status': STATUS_OK
            }

        time_now = time.time()

        trials = Trials()
        algo = tpe.suggest if use_BOA else rand.suggest
        # algo = partial(algo, n_startup_jobs=1) 
        best_hp = fmin(_evaluate_hp, self.hp_space, algo=algo, max_evals=max_evals, 
                       trials=trials, rstate=np.random.default_rng(1), verbose=True)

        spent_time = (time.time() - time_now) 
        logger.info(f'Combiner._optimize_HP() : {spent_time:.4f} sec elapsed')

        return best_hp, trials


    def train(self):
        # Train combiner model -------------------------------------
        time_now = time.time()

        train_dataset, train_loader = get_data_provider(self.configs, flag='ensemble_train', step_by_step=True)

        basemodel_preds = np.empty((len(self.basemodels), len(train_loader)))
        basemodel_losses = np.empty((len(self.basemodels), len(train_loader)))
        criterion = self._select_criterion()

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for m, basemodel in enumerate(self.basemodels):
                basemodel_losses[m, t], basemodel_preds[m, t] = basemodel.proceed_onestep(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, 
                    training=True) # NOTE : Base models ARE trained in ensemble_training period.
            _mem_util.print_memory_usage()

        spent_time = (time.time() - time_now) 
        logger.info(f"CombinerModel.train() : {spent_time:.4f} sec elapsed for getting base models' predictions")

        # Hyperparameter Optimization -------------------------------------
        hpo_peroid = min(self.MAX_HPO_EVAL_PEROID, len(train_loader))
        self.basemodel_losses = basemodel_losses[:, -hpo_peroid:]
        self.truths = train_dataset.data_y[-hpo_peroid:, -1]
        hp_boa, trials_boa = self._optimize_HP(max_evals=self.configs.max_hpo_eval)

        report.plot_hpo_result(hp_boa, trials_boa, "HyperParameter Optimization for Combiner",
                            self._get_result_path()+"/hpo_result.pdf")
    
        self.hp_dict = hp_boa
        

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training: bool = True):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        basemodel_preds = np.empty((len(self.basemodels)))
        basemodel_losses = np.empty((len(self.basemodels), 1))

        for m, basemodel in enumerate(self.basemodels):
            basemodel_preds[m], basemodel_losses[m, 0] = basemodel.proceed_onestep(
                batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training) 

        basemodel_losses = np.concatenate((self.basemodel_losses, basemodel_losses), axis=1)
        lookback_window_size = int(self.hp_dict['lookback_window_size'])
        basemodel_weights = self.compute_basemodel_weights(
            self.hp_dict, basemodel_losses[:, -lookback_window_size:], self.basemodel_weights)
        self.basemodel_weights = basemodel_weights

        # Adaptive HPO 
        if self.configs.adaptive_hpo:
            hpo_peroid = min(self.MAX_HPO_EVAL_PEROID, basemodel_losses.shape[1])
            self.basemodel_losses = basemodel_losses[:, -hpo_peroid:]
            self.truths = np.concatenate((self.truths, batch_y[-1:, -1, -1]))[-hpo_peroid:]
            self.hpo_counter += 1
            if self.hpo_counter == self.configs.hpo_interval:
                self.hp_dict, _ = self._optimize_HP(max_evals=self.configs.max_hpo_eval)
                self.hpo_counter = 0                

        y_hat = np.dot(basemodel_weights, basemodel_preds)
        return y_hat, basemodel_preds


    # 
    # NOTE:
    # In test() func, 'Adaptive HPO' is not applied. 
    #
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
        basemodel_losses = np.concatenate((self.basemodel_losses, basemodel_losses), axis=1)

        lookback_window_size = int(self.hp_dict['lookback_window_size'])
        basemodel_weights = None
        for t in range(len(y)):
            basemodel_weights = self.compute_basemodel_weights(
                self.hp_dict, basemodel_losses[:, -lookback_window_size:], basemodel_weights)
            y_hat[t] = np.dot(basemodel_weights, basemodel_preds[:, t:t+1])
            weights_hist[t] = basemodel_weights

        spent_time = (time.time() - time_now) 
        logger.info(f'CombinerModel.test() : {spent_time:.4f}sec elapsed for testing')

        report.plot_weights(weights_hist, "Base Model Weights",
                           self._get_result_path() + "/basemodel_weights.pdf")

        if need_to_invert_data:
            n_features = test_data.data_y.shape[1]
            data_y = np.zeros((len(y), n_features))
            data_y_hat = np.zeros((len(y), n_features))
            data_y[:, -1] = y
            data_y_hat[:, -1] = y_hat
            y = test_data.inverse_transform(data_y)[:, -1]
            y_hat = test_data.inverse_transform(data_y_hat)[:, -1]

        losses = criterion(torch.tensor(y_hat), y)

        logger.info(f"CombinerModel.test() : Loss ----- ")
        logger.info(f"max={np.max(losses):.6f}, mean={np.mean(losses):.6f}, min={np.min(losses):.6f}, var={np.var(losses):.6f})")

        report.plot_forecast(y, y_hat, "Combiner Forecast", 
                            self._get_result_path() + "/combiner_forecast.pdf")

        return y, y_hat, losses

