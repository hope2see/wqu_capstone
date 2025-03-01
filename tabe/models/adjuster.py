
# Gaussian Process Model 
# ref) https://pyro.ai/examples/gp.html

import os
import time
import numpy as np
import torch

from scipy.stats import norm

import pyro
import pyro.contrib.gp as gp

from hyperopt import hp, tpe, rand, fmin, Trials, STATUS_OK

from tabe.data_provider.dataset_loader import get_data_provider
from tabe.models.abstractmodel import AbstractModel
from tabe.utils.mem_util import MemUtil
from tabe.utils.misc_util import logger
import tabe.utils.report as report


smoke_test = "CI" in os.environ  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.9.1')
pyro.set_rng_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)


_mem_util = MemUtil(rss_mem=False, python_mem=False)


class AdjusterModel(AbstractModel):
    # Maximum lookback-window size for fitting gaussian process model
    MIN_LOOKBACK_WIN = 10
    MAX_LOOKBACK_WIN = 50

    # Period for HPO (HyperParameter Optimization)
    # The most recent period of this length is used for HPO. 
    # For the first HPO, the ensemble-training period can be used. 
    # But, to provide continuous adaptive HPO feature, 
    # the latest input should be used (or added) for HPO. 
    # And, ever-growing input size is not practically acceptable. 
    # Thus, we use fixed-size evaluation period for HPO.
    MAX_EVAL_PEROID = MAX_LOOKBACK_WIN * 4


    def __init__(self, configs, combiner_model):
        super().__init__(configs, "Adjuster")
        self.combiner_model = combiner_model # must've been trained already
        self._set_gpm_kernel()
        self.gpm = None # Gaussain Process Model 
        self.hp_space = {
            'lookback_window_size': hp.quniform('lookback_window_size', self.MIN_LOOKBACK_WIN, self.MAX_LOOKBACK_WIN, 2),
            'noise': hp.uniform('noise', 0.0, 0.4),
        }
        self.hp_dict = None   # currently active hyper-parameters
        self.hpo_counter = 0 # used for counting timesteps for Adaptive HPO
        self.y_hat_cbm = None # the last predictions of combiner. shape = (HPO_EVALUATION_PEROID)
        self.truths = None # the last 'y' values. shape = (HPO_EVALUATION_PEROID)


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
            self.kernel = kernel_class(input_dim=input_dim, variance=variance, lengthscale=lengthscale)
        else:
            self.kernel = kernel_class(input_dim=input_dim, variance=variance)


    def _train_new_gpmodel(self, hp_dict, y):
        pyro.clear_param_store() # NOTE : Need to do everytime? 

        lookback_window_size = int(hp_dict['lookback_window_size'])
        noise = int(hp_dict['noise'])
        if len(y) > lookback_window_size+1:
            y = y[-(lookback_window_size+1):]
        X = y[:-1]
        y = y[1:]
        gpm = gp.models.GPRegression(X, y, self.kernel, 
                                            noise=torch.tensor(noise), mean_function=None, jitter=1e-6)
        gpm.set_data(X, y)
        self.optimizer = torch.optim.Adam(gpm.parameters(), lr=0.005)
        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        gp.util.train(gpm, self.optimizer, self.loss_fn, num_steps=self.configs.max_gp_opt_steps)
        return gpm


    def _forward_onestep(self, hp_dict, gpm, y):
        assert y.shape[0]==1, "Allowed to add only one observation.."
        # pyro.clear_param_store() # NOTE : Need to do everytime? 

        # incorporate new observation(s)
        X = torch.cat([gpm.X, gpm.y[-1:]]) # Add the last y to the end of 'X's
        y = torch.cat([gpm.y, y]) # Add new observation to the end of 'y's

        lookback_window_size = int(hp_dict['lookback_window_size'])
        if len(X) > lookback_window_size:
            X = X[-lookback_window_size:]
            y = y[-lookback_window_size:]
        gpm.set_data(X, y)
        gp.util.train(gpm, self.optimizer, self.loss_fn, num_steps=self.configs.max_gp_opt_steps)
        return gpm


    def _predict_next(self, gpm):
        last_deviation = gpm.y[-1:] 
        with torch.no_grad():
            exp_deviation, cov = gpm(last_deviation, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  
        # print(f"Adjuster.proceed_onestep(): exp_dev={exp_deviation.item():.6f}, sd_dev={sd.item():.6f}")
        return exp_deviation, sd


    def _optimize_HP(self, use_BOA=False, max_evals=10):

        # Objective function (loss function) for hyper-parameter optimization
        # Loss == Mean of the lossees in all timesteps in the period [lookback_window_size, len(y)]
        def _evaluate_hp(hp_dict):
            deviations = (self.truths - self.y_hat_cbm)[:-1] # exclude the last one, becuause it is the target to predict
            assert len(deviations) > self.MAX_LOOKBACK_WIN
            t = self.MAX_LOOKBACK_WIN # We should start from the same timestep to evaluate the same period.
            gpm = None            
            losses = []
            while t < len(deviations):
                if gpm is None:
                    gpm = self._train_new_gpmodel(hp_dict, torch.Tensor(deviations[:t]))
                else:
                    gpm = self._forward_onestep(hp_dict, gpm, torch.tensor([deviations[t]]))
                exp_deviation, _ = self._predict_next(gpm)
                next_y_hat = self.y_hat_cbm[t] + exp_deviation
                next_y = self.truths[t]
                losses.append(np.abs(next_y_hat - next_y))
                t += 1
            mean_loss = np.mean(losses)
            var_loss = np.var(losses)

            return {
                'loss': mean_loss,         
                'loss_variance': var_loss, 
                'status': STATUS_OK
            }

        time_now = time.time()

        trials = Trials()
        algo = tpe.suggest if use_BOA else rand.suggest
        best_hp = fmin(_evaluate_hp, self.hp_space, algo=algo, max_evals=max_evals, 
                       trials=trials, rstate=np.random.default_rng(1), verbose=True)

        spent_time = (time.time() - time_now) 
        logger.info(f'Adjuster._optimize_HP() : {spent_time:.4f} sec elapsed')
        report.print_dict(best_hp, '[ Adjuster HP ]')

        return best_hp, trials


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

        eval_peroid = min(self.MAX_EVAL_PEROID, len(train_loader)-1)
        self.y_hat_cbm = y_hat_cbm[-eval_peroid:]
        self.truths = y[-eval_peroid:]
        if self.configs.adaptive_hpo:
            hp_boa, trials_boa = self._optimize_HP(max_evals=self.configs.max_hpo_eval)
            report.plot_hpo_result(trials_boa, "HyperParameter Optimization for Adjuster",
                                self._get_result_path()+"/hpo_result.pdf")
            self.hp_dict = hp_boa
            logger.info(f"Adjuster HPO : lookback_window_size = {hp_boa['lookback_window_size']}")
        else:
            self.hp_dict = {
                'lookback_window_size':self.configs.gpm_lookback_win,
                'noise':self.configs.gpm_noise
                }

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

        # adjust combinerModel's prediction by adding expected deviation 
        y_hat = y_hat_cbm + pred_deviation
        z_val = norm.ppf(self.configs.quantile)  # assuming Gaussian distribution
        y_hat_quantile_low = y_hat - devi_stddev * z_val
        y_hat_quantile_high = y_hat + devi_stddev * z_val

        # calculate the actuall loss of next timestep
        y = batch_y[0, -1, -1] 

        if training:
            true_deviation = torch.tensor([y.item() - y_hat_cbm])
            self.gpm = self._forward_onestep(self.hp_dict, self.gpm, true_deviation)

        # Adaptive HPO 
        if self.configs.adaptive_hpo:
            self.y_hat_cbm = np.concatenate((self.y_hat_cbm, np.array([y_hat_cbm])))
            self.truths = np.concatenate((self.truths, np.array([y])))
            if len(self.y_hat_cbm) > self.MAX_EVAL_PEROID: 
                self.y_hat_cbm = self.y_hat_cbm[:, -self.MAX_EVAL_PEROID:]
                self.truths = self.truths[:, -self.MAX_EVAL_PEROID:]
            self.hpo_counter += 1
            if self.hpo_counter == self.configs.hpo_interval:
                self.hp_dict, _ = self._optimize_HP(max_evals=self.configs.max_hpo_eval)
                self.hpo_counter = 0                

        return y_hat, y_hat_cbm, y_hat_bsm, y_hat_quantile_low, y_hat_quantile_high


    def test(self):
        test_set, test_loader = get_data_provider(self.configs, flag='test', step_by_step=True)
        y = test_set.data_y[self.configs.seq_len:, -1]
        need_to_invert_data = True if (test_set.scale and self.configs.inverse) else False

        y_hat = np.empty_like(y)
        y_hat_cbm = np.empty_like(y)
        y_hat_bsm = np.empty((len(self.combiner_model.basemodels), len(y)))
        y_hat_q_low = np.empty_like(y)
        y_hat_q_high = np.empty_like(y)

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            y_hat[t], y_hat_cbm[t], y_hat_bsm[:,t], y_hat_q_low[t], y_hat_q_high[t] = \
                self.proceed_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark, training=True)            
            _mem_util.print_memory_usage()

        report.plot_gpmodel(self.gpm, filepath=self._get_result_path()+"/gpmodel_analysis.pdf")

        if need_to_invert_data:
            n_features = test_set.data_y.shape[1]
            data_y = np.zeros((len(y), n_features))
            data_y_hat = np.zeros((len(y), n_features))
            data_y_hat_q_low = np.zeros((len(y), n_features))
            data_y_hat_q_high = np.zeros((len(y), n_features))
            data_y_hat_cbm = np.zeros((len(y), n_features))
            data_y[:, -1] = y
            data_y_hat[:, -1] = y_hat
            data_y_hat_q_low[:, -1] = y_hat_q_low
            data_y_hat_q_high[:, -1] = y_hat_q_high
            data_y_hat_cbm[:, -1] = y_hat_cbm
            y = test_set.inverse_transform(data_y)[:, -1]
            y_hat = test_set.inverse_transform(data_y_hat)[:, -1]
            y_hat_q_low = test_set.inverse_transform(data_y_hat_q_low)[:, -1]
            y_hat_q_high = test_set.inverse_transform(data_y_hat_q_high)[:, -1]
            y_hat_cbm = test_set.inverse_transform(data_y_hat_cbm)[:, -1]
            for i in range(len(y_hat_bsm)):
                data_y_hat_bsm = np.zeros((len(y), n_features))
                data_y_hat_bsm[:, -1] = y_hat_bsm[i]
                y_hat_bsm[i] = test_set.inverse_transform(data_y_hat_bsm)[:, -1]

        return y, y_hat, y_hat_cbm, y_hat_bsm, y_hat_q_low, y_hat_q_high
