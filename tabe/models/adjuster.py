
# Gaussian Process Model 
# ref) https://pyro.ai/examples/gp.html

import os
import io 
import time
import numpy as np
import torch

from scipy.stats import norm

import pyro
import pyro.contrib.gp as gp

from hyperopt import hp, tpe, rand, fmin, Trials, STATUS_OK

from utils.metrics import MAE, MSE, RMSE, MAPE, MSPE

from tabe.data_provider.dataset_loader import get_data_provider
from tabe.models.abstractmodel import AbstractModel
from tabe.utils.mem_util import MemUtil
from tabe.utils.misc_util import logger, OptimTracker
import tabe.utils.weighting as weighting
import tabe.utils.report as report

smoke_test = "CI" in os.environ  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.9.1')
pyro.set_rng_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)


_mem_util = MemUtil(rss_mem=False, python_mem=False)


class AdjusterModel(AbstractModel):
    # lookback-window size for fitting gaussian process model
    # MIN_LOOKBACK_WIN = 10
    # MAX_LOOKBACK_WIN = 80
    UNLIMITED_LOOKBACK_WIN = -1

    # Period for HPO (HyperParameter Optimization)
    # The most recent period of this length is used for HPO. 
    # For the first HPO, the ensemble-training period can be used. 
    # But, to provide continuous adaptive HPO feature, 
    # the latest input should be used (or added) for HPO. 
    # And, ever-growing input size is not practically acceptable. 
    # Thus, we use fixed-size evaluation period for HPO.
    #
    # |<-  MAX_LOOKBACK_WIN   ->|<-     HPO_EVAL_PEROID      ->|
    # [0,1,..              ,t-1][t,t+1,... ,t+hpo_eval_period-1]
    HPO_EVAL_PEROID = 20 # Probably, the more, the better. But, too mush time cost. 


    def __init__(self, configs, combiner_model):
        super().__init__(configs, "Adjuster")
        self.combiner_model = combiner_model # must've been trained already
        self.gpm = None # Gaussain Process Model 
        self._set_gpm_kernel()
        self.gpm_noise = self.configs.gpm_noise
        self.y_hat = np.array([])
        self.y_hat_cbm = None # the last predictions of combiner. shape = (HPO_EVALUATION_PEROID)
        self.truths = None # the last 'y' values. shape = (HPO_EVALUATION_PEROID)
        self.credibility = 0.5 # relative credibility against combiner model. Initially neutral value 0.5
        self.hpo_policy = self.configs.hpo_policy
        if self.hpo_policy == 0 :
            self.hp_dict = {
                'gpm_lookback_win':self.configs.gpm_lookback_win,
                # 'lookback_window':self.configs.lookback_win, # for weighting 
                # 'discount_factor':self.configs.discount_factor,
                # 'avg_method':self.configs.avg_method,
                # 'weighting_method':self.configs.weighting_method,
                # 'scaling_factor':self.configs.scaling_factor,
                # 'smoothing_factor':self.configs.smoothing_factor
                }
        else:
            self.hp_space = {
                'gpm_lookback_win': hp.choice('gpm_lookback_win', [5, 10, 30, 50, 100, self.UNLIMITED_LOOKBACK_WIN]), 
                # 'lookback_window': hp.quniform('lookback_window', 1, 10, 1),
                # 'discount_factor':hp.uniform('discount_factor', 1.0, 2.0),
                # 'avg_method': hp.choice('avg_method', [weighting.AvgMethod.MEAN, weighting.AvgMethod.MEAN_SQUARED]),
                # 'weighting_method': hp.choice('weighting_method', 
                #                         [weighting.WeightingMethod.INVERTED, weighting.WeightingMethod.SQUARED_INVERTED, weighting.WeightingMethod.SOFTMAX]),
                # 'scaling_factor':hp.choice('scaling_factor', [10, 30, 50, 100]),
                # 'smoothing_factor': hp.uniform('smoothing_factor', 0.0, 0.2),
            }
            self.hp_dict = None
            self.hpo_counter = 0 # used for counting timesteps for Adaptive HPO


    def _set_gpm_kernel(self):
        kernels = {
            # name      : (class, need_lengthscale)
            'RBF'       : (gp.kernels.RBF, True),
            'Matern32'  : (gp.kernels.Matern32, True),
            'Matern52'  : (gp.kernels.Matern52, True),
            'Linear'    : (gp.kernels.Linear, False),
            'Brownian'  : (gp.kernels.Brownian, False),
        }
        input_dim = 1
        variance = torch.tensor(1)
        lengthscale = torch.tensor(1.5)
        kernel_name = self.configs.gpm_kernel
        kernel_class = kernels[kernel_name][0]
        need_lengthscale = kernels[kernel_name][1]
        if need_lengthscale:
            self.gpm_kernel = kernel_class(input_dim=input_dim, variance=variance, lengthscale=lengthscale)
        else:
            self.gpm_kernel = kernel_class(input_dim=input_dim, variance=variance)


    def _train_new_gpmodel(self, hp_dict, y):
        pyro.clear_param_store() # NOTE : Need to do everytime? 

        gpm_lookback_win = int(hp_dict['gpm_lookback_win'])
        if gpm_lookback_win != self.UNLIMITED_LOOKBACK_WIN and len(y) > gpm_lookback_win+1:
            y = y[-(gpm_lookback_win+1):]
        X = y[:-1]
        y = y[1:]
        gpm = gp.models.GPRegression(X, y, self.gpm_kernel, 
                                            noise=torch.tensor(self.gpm_noise), mean_function=None, jitter=1e-6)
        gpm.set_data(X, y)
        self.optimizer = torch.optim.Adam(gpm.parameters(), lr=0.005)
        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        
        optim_tracker = OptimTracker(use_early_stop=True, patience=self.configs.max_gp_opt_patience, verbose=False, save_to_file=False)
        num_batch = 10
        for step in range(1, self.configs.max_gp_opt_steps, num_batch):
            loss = gp.util.train(gpm, self.optimizer, self.loss_fn, num_steps=num_batch)
            mean_loss = np.mean(loss).item() / len(y) # mean loss for one-step
            optim_tracker(mean_loss, None)
            if optim_tracker.early_stop:
                break
        logger.debug(f"Adj.HPO: when lb_win_size={gpm_lookback_win}, after training {step} times, loss={mean_loss:.4f}")
        
        return gpm
 

    def _forward_onestep(self, hp_dict, gpm, y):
        assert y.shape[0]==1, "Allowed to add only one observation.."
        # pyro.clear_param_store() # NOTE : Need to do everytime? 

        # incorporate new observation(s)
        X = torch.cat([gpm.X, gpm.y[-1:]]) # Add the last y to the end of 'X's
        y = torch.cat([gpm.y, y]) # Add new observation to the end of 'y's

        gpm_lookback_win = int(hp_dict['gpm_lookback_win'])
        if gpm_lookback_win != self.UNLIMITED_LOOKBACK_WIN and len(y) > gpm_lookback_win+1:
            X = X[-gpm_lookback_win:]
            y = y[-gpm_lookback_win:]
        gpm.set_data(X, y)

        optim_tracker = OptimTracker(use_early_stop=True, patience=self.configs.max_gp_opt_patience, verbose=False, save_to_file=False)
        num_batch = 10
        for step in range(1, self.configs.max_gp_opt_steps, num_batch):
            loss = gp.util.train(gpm, self.optimizer, self.loss_fn, num_steps=num_batch)
            mean_loss = np.mean(loss).item() / len(y) # mean loss for one timestep
            optim_tracker(mean_loss, None)
            if optim_tracker.early_stop:
                break                
        logger.debug(f"Adj.HPO: when lb_win_size={gpm_lookback_win}, after training  {step} times, loss={mean_loss:.4f}")
        return gpm


    def _predict_next(self, gpm):
        last_deviation = gpm.y[-1:] 
        with torch.no_grad():
            exp_deviation, cov = gpm(last_deviation, full_cov=False, noiseless=False)
        sd = cov.diag().sqrt()  
        logger.info(f"Adjuster._predict_next(): exp_dev={exp_deviation.item():.6f}, sd_dev={sd.item():.6f}")
        return exp_deviation, sd


    def _optimize_HP(self, 
                     search_alg=0, # 0 : ad-hoc, 1 : BOA, 2 : random
                     max_evals=10):
        
        # Objective function (loss function) for hyper-parameter optimization
        # Loss == Mean of the lossees in all timesteps in the period [gpm_lookback_win, len(y)]
        def _evaluate_hp(hp_dict):
            deviations = (self.truths - self.y_hat_cbm)[:-1] # exclude the last one, becuause it is the target to predict
            assert len(deviations) > self.HPO_EVAL_PEROID
            gpm = None            
            losses = []
            for t in range(len(deviations) - self.HPO_EVAL_PEROID, len(deviations)):
                if gpm is None:
                    gpm = self._train_new_gpmodel(hp_dict, torch.Tensor(deviations[:t]))
                else:
                    gpm = self._forward_onestep(hp_dict, gpm, torch.tensor([deviations[t]]))
                exp_deviation, _ = self._predict_next(gpm)
                next_y_hat = self.y_hat_cbm[t] + exp_deviation
                next_y = self.truths[t]
                losses.append(np.abs(next_y_hat - next_y)) # MAE!  # Need to use the loss metric of configuration 
                t += 1
            mean_loss = np.mean(losses)
            var_loss = np.var(losses)
            self.optim_tracker(mean_loss, hp_dict)

            return {
                'loss': mean_loss,         
                'loss_variance': var_loss, 
                'status': STATUS_OK
            }

        time_now = time.time()

        self.optim_tracker = OptimTracker(use_early_stop=False, patience=self.configs.max_gp_opt_patience, verbose=True, save_to_file=False)
        trials = Trials()
        algo = tpe.suggest if search_alg == 1 else rand.suggest
        self.best_hp = fmin(_evaluate_hp, self.hp_space, algo=algo, max_evals=max_evals, 
                trials=trials, rstate=np.random.default_rng(1), verbose=True)

        spent_time = (time.time() - time_now) 

        logger.info(f'Adjuster._optimize_HP() : {spent_time:.4f}sec elapsed. min_loss={self.optim_tracker.val_loss_min:.5f}')            
        report.print_dict(self.best_hp, '[ Adjuster HP ]')
        return self.best_hp, trials


    def train(self):
        # NOTE
        # For training adjusrter model , we use the data in validation period of the base models.
        # The data is also used for training combiner model, it doesn't matter because... TODO
        train_dataset, train_loader = get_data_provider(self.configs, flag='ensemble_train', step_by_step=True)
        y = train_dataset.data_y[self.configs.seq_len:, -1]
        assert len(y) == len(train_loader)

        y_hat_cbm = np.empty_like(y)

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            y_hat_t, _ = self.combiner_model.proceed_onestep(
                batch_x, batch_y, batch_x_mark, batch_y_mark)
            y_hat_cbm[t] = y_hat_t
            _mem_util.print_memory_usage()

        assert self.HPO_EVAL_PEROID <= len(train_loader), \
                    f'length of train data ({len(train_loader)}) should be longer than HPO_EVAL_PEROID({self.HPO_EVAL_PEROID})'     

        self.y_hat_cbm = y_hat_cbm
        self.truths = y

        if self.hpo_policy != 0: 
            self.hp_dict, trials = self._optimize_HP(search_alg=1, max_evals=self.configs.max_hpo_eval)
            if trials is not None:
                report.plot_hpo_result(trials, "HyperParameter Optimization for Adjuster",
                                self._get_result_path()+"/hpo_result.pdf")

        # Gaussian Process model is to used to predict next deviation from past predctions
        # We exclude the last deviation for training, becuause that is the actual target. 
        deviations = (self.truths - self.y_hat_cbm)[:-1]
        self.gpm = self._train_new_gpmodel(self.hp_dict, torch.Tensor(deviations))


    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        # estimate the next deviation with the last deviation 
        pred_deviation, devi_stddev = self._predict_next(self.gpm)

        # get combiner model's predition
        y_hat_cbm, y_hat_bsm = self.combiner_model.proceed_onestep(
            batch_x, batch_y, batch_x_mark, batch_y_mark, training)                
        
        # get credibility (or weight over combiner) of Adjuster model 
        eval_period = min(len(self.y_hat), self.configs.lookback_win)
        if eval_period > 0:
            my_loss = np.abs(self.y_hat[-eval_period:] - self.truths[-eval_period:])   
            cbm_loss = np.abs(self.y_hat_cbm[-eval_period:] - self.truths[-eval_period:])
            model_losses = np.array([my_loss, cbm_loss])
            prev_weights = np.array([self.credibility, 1.0-self.credibility])
            weights = weighting.compute_model_weights(model_losses, prev_weights, 
                                    lookback_window=eval_period, 
                                    discount_factor=self.configs.discount_factor, 
                                    avg_method=self.configs.avg_method, 
                                    weighting_method=self.configs.weighting_method,
                                    softmax_scaling_factor=self.configs.scaling_factor, 
                                    smoothing_factor=self.configs.smoothing_factor)
            self.credibility = weights[0]

            logger.debug("Adj.predict : Adj Losses : " + "[" + ", ".join(f'{l:.5f}' for l in my_loss) + "]")
            logger.debug("Adj.predict : Cbm Losses : " + "[" + ", ".join(f'{l:.5f}' for l in cbm_loss) + "]")
            logger.info(f'Adj.predict : adjuster_credibility = {self.credibility:.5f}')

        y_hat = y_hat_cbm + pred_deviation.item()
                     
        # use_choice_policy = False # TODO ! 
        # if use_choice_policy: 
        #     final_pred = y_hat if self.credibility > 0.5 else y_hat_cbm        
        # adjust combinerModel's prediction by adding expected deviation 
        final_pred = y_hat_cbm + (pred_deviation.item() * self.credibility)

        # calculate the actuall loss of next timestep
        y = batch_y[0, -1, -1] 

        self.y_hat_cbm = np.concatenate((self.y_hat_cbm, np.array([y_hat_cbm])))
        self.y_hat = np.concatenate((self.y_hat, np.array([y_hat])))
        self.truths = np.concatenate((self.truths, np.array([y])))

        if training:
            true_deviation = torch.tensor([y.item() - y_hat_cbm])
            self.gpm = self._forward_onestep(self.hp_dict, self.gpm, true_deviation)

        # Adaptive HPO 
        if self.configs.hpo_policy == 2: 
            self.hpo_counter += 1
            if self.hpo_counter == self.configs.hpo_interval:
                self.hp_dict, _ = self._optimize_HP(search_alg=1, max_evals=self.configs.max_hpo_eval)
                self.hpo_counter = 0                

        # logger.info(f'Adj.predict : final_pred={final_pred:.5f}, y_hat={y_hat:.5f}, y_hat_cbm={y_hat_cbm:.5f}')
        return final_pred, y_hat, y_hat_cbm, y_hat_bsm, devi_stddev


    def test(self):
        test_set, test_loader = get_data_provider(self.configs, flag='test', step_by_step=True)
        y = test_set.data_y[self.configs.seq_len:, -1]
        need_to_invert_data = True if (test_set.scale and self.configs.inverse) else False

        final_pred = np.empty_like(y)
        y_hat = np.empty_like(y)
        y_hat_cbm = np.empty_like(y)
        y_hat_bsm = np.empty((len(self.combiner_model.basemodels), len(y)))
        y_hat_q_low = np.empty_like(y)
        y_hat_q_high = np.empty_like(y)
        devi_stddev = np.empty_like(y)

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            final_pred[t], y_hat[t], y_hat_cbm[t], y_hat_bsm[:,t], devi_stddev[t] = \
                self.proceed_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark, training=True)            
            _mem_util.print_memory_usage()

        report.plot_gpmodel(self.gpm, filepath=self._get_result_path()+"/gpmodel_analysis.pdf")

        z_val = norm.ppf(self.configs.quantile) 
        y_hat_q_low = y_hat - devi_stddev * z_val
        y_hat_q_high = y_hat + devi_stddev * z_val

        z_val = norm.ppf(self.configs.buy_threshold_prob) 
        buy_threshold_q = y_hat - devi_stddev * z_val

        if need_to_invert_data:
            n_features = test_set.data_y.shape[1]
            data_y = np.zeros((len(y), n_features))
            data_final_pred = np.zeros((len(y), n_features))
            data_y_hat_q_low = np.zeros((len(y), n_features))
            data_y_hat_q_high = np.zeros((len(y), n_features))
            data_buy_threshold_q = np.zeros((len(y), n_features))
            data_y_hat_cbm = np.zeros((len(y), n_features))
            data_y[:, -1] = y
            data_final_pred[:, -1] = final_pred
            data_y_hat_q_low[:, -1] = y_hat_q_low
            data_y_hat_q_high[:, -1] = y_hat_q_high
            data_buy_threshold_q[:, -1] = buy_threshold_q
            data_y_hat_cbm[:, -1] = y_hat_cbm
            y = test_set.inverse_transform(data_y)[:, -1]
            final_pred = test_set.inverse_transform(data_final_pred)[:, -1]
            y_hat_q_low = test_set.inverse_transform(data_y_hat_q_low)[:, -1]
            y_hat_q_high = test_set.inverse_transform(data_y_hat_q_high)[:, -1]
            buy_threshold_q = test_set.inverse_transform(data_buy_threshold_q)[:, -1]
            y_hat_cbm = test_set.inverse_transform(data_y_hat_cbm)[:, -1]
            for i in range(len(y_hat_bsm)):
                data_y_hat_bsm = np.zeros((len(y), n_features))
                data_y_hat_bsm[:, -1] = y_hat_bsm[i]
                y_hat_bsm[i] = test_set.inverse_transform(data_y_hat_bsm)[:, -1]

        return y, final_pred, y_hat_cbm, y_hat_bsm, y_hat_q_low, y_hat_q_high, buy_threshold_q, devi_stddev
