
# Gaussian Process Model 

# ref) https://pyro.ai/examples/gp.html

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from statsmodels.tsa.statespace.sarimax import SARIMAX

from dataset_loader import get_data_provider
from combiner import CombinerModel
from mem_util import MemUtil


smoke_test = "CI" in os.environ  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.9.1')
pyro.set_rng_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)


class AdjusterModel(object):
    FITTING_WINDOW = 100 # TODO 
    MAX_OPT_STEPS = 3000 # maximum steps of optimization

    def __init__(self, configs, combiner_model, gpm_kernel=None):
        self.configs = configs
        self.combiner_model = combiner_model # must've been trained already
        if gpm_kernel is None:
            gpm_kernel = gp.kernels.RBF(
                # TODO : Optimize HP 
                input_dim=1, variance=torch.tensor(1), lengthscale=torch.tensor(1.5)
            )
        self.kernel = gpm_kernel 
        self.gpm = None # Gaussain Process Model 
        self.dataset = {}
        self.dataloader = {}
        self.last_y = None

    def _get_data(self, flag):
        if flag not in self.dataset:
            self.dataset[flag], self.dataloader[flag] = get_data_provider(self.configs, flag)
        return self.dataset[flag], self.dataloader[flag]

    # NOTE: TBD
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _train_gpmodel(self, y):
        pyro.clear_param_store() # NOTE : Need to do everytime? 

        # TODO : HP optimization 
        if self.gpm is None:
            X = y[:-1]
            y = y[1:]
            self.gpm = gp.models.GPRegression(X, y, self.kernel, 
                                              noise=torch.tensor(0.2), mean_function=None, jitter=1e-6)
            self.optimizer = torch.optim.Adam(self.gpm.parameters(), lr=0.005)
            self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        else:
            assert y.shape[0]==1, "Allowed to add only one observation.."
            # incorporate new observation(s)
            X = torch.cat([self.gpm.X, self.gpm.y[-1:]]) # Add the last y to the end of 'X's
            y = torch.cat([self.gpm.y, y]) # Add new observation to the end of 'y's
            self.gpm.set_data(X, y)

        # if len(X) > self.configs.seq_len: 
        #     X = X[-self.configs.seq_len:]
        #     y = y[-self.configs.seq_len:]
        if len(X) > self.FITTING_WINDOW: 
            X = X[-self.FITTING_WINDOW:]
            y = y[-self.FITTING_WINDOW:]
            self.gpm.set_data(X, y)

        gp.util.train(self.gpm, self.optimizer, self.loss_fn, num_steps=self.MAX_OPT_STEPS)


    def _predict_next(self):
        last_deviation = self.gpm.y[-1:] 
        with torch.no_grad():
            exp_deviation, cov = self.gpm(last_deviation, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  
        print(f"Adjuster.proceed_onestep(): exp_dev={exp_deviation.item():.6f}, sd_dev={sd.item():.6f}")
        return exp_deviation


    def train(self):
        # NOTE
        # For training adjusrter model , we use the data in validation period of the base models.
        # The data is also used for training combiner model, it doesn't matter because... TODO
        train_dataset, train_loader = get_data_provider(self.configs, flag='ensemble_train', step_by_step=True)
        y = train_dataset.data_y[self.configs.seq_len:, -1]
        assert len(y) == len(train_loader)

        y_hat_cbm = np.empty_like(y)
        criterion = self._select_criterion()

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            y_hat_t, _ = self.combiner_model.proceed_onestep(
                batch_x, batch_y, batch_x_mark, batch_y_mark, criterion)
            y_hat_cbm[t] = y_hat_t
            MemUtil.print_memory_usage()

        # Gaussian Process model is to used to predict next deviation from past predctions
        deviations = y - y_hat_cbm
        self._train_gpmodel(torch.Tensor(deviations))


    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        # estimate the next deviation with the last deviation 
        pred_deviation = self._predict_next()

        # get combiner model's predition
        y_hat_cbm, y_hat_bsm = self.combiner_model.proceed_onestep(
            batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training)                

        # adjust combinerModel's prediction by adding expected deviation 
        y_hat = y_hat_cbm + pred_deviation

        # calculate the actuall loss of next timestep
        y = batch_y[0, -1, -1] 
        # loss = criterion(torch.tensor(y_hat), y).item()

        if training:
            true_deviation = torch.tensor([y.item() - y_hat_cbm])
            self._train_gpmodel(true_deviation)

        return y_hat, y_hat_cbm, y_hat_bsm


    def test(self):
        test_set, test_loader = get_data_provider(self.configs, flag='test', step_by_step=True)
        y = test_set.data_y[self.configs.seq_len:, -1]
        need_to_invert_data = True if (test_set.scale and self.configs.inverse) else False

        y_hat = np.empty_like(y)
        y_hat_cbm = np.empty_like(y)
        y_hat_bsm = np.empty((len(self.combiner_model.basemodels), len(y)))

        criterion = self._select_criterion()

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            y_hat[t], y_hat_cbm[t], y_hat_bsm[:,t] = \
                self.proceed_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training=True)            
            MemUtil.print_memory_usage()

        if need_to_invert_data:
            n_features = test_set.data_y.shape[1]
            data_y = np.zeros((len(y), n_features))
            data_y_hat = np.zeros((len(y), n_features))
            data_y_hat_cbm = np.zeros((len(y), n_features))
            data_y[:, -1] = y
            data_y_hat[:, -1] = y_hat
            data_y_hat_cbm[:, -1] = y_hat_cbm
            y = test_set.inverse_transform(data_y)[:, -1]
            y_hat = test_set.inverse_transform(data_y_hat)[:, -1]
            y_hat_cbm = test_set.inverse_transform(data_y_hat_cbm)[:, -1]
            for i in range(len(y_hat_bsm)):
                data_y_hat_bsm = np.zeros((len(y), n_features))
                data_y_hat_bsm[:, -1] = y_hat_bsm[i]
                y_hat_bsm[i] = test_set.inverse_transform(data_y_hat_bsm)[:, -1]

        return y, y_hat, y_hat_cbm, y_hat_bsm


    def plot_gpmodel(self, plot_observed_data=True, plot_predictions=True, n_test=500, x_range=None):
        if x_range is None:
            min = self.gpm.X.numpy().min()
            max = self.gpm.X.numpy().max()
            x_range = (min - abs(max-min)*0.1, max + abs(max-min)*0.1)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Prediction Deviation Distribution (Analyzed in Gaussian Process)")
        ax.set_ylabel("Prediction deviation at day t")
        ax.set_xlabel("Prediction deviation at day t-1")

        if plot_observed_data:
            ax.plot(self.gpm.X.numpy(), self.gpm.y.numpy(), "kx", label="observations")

        if plot_predictions:
            Xtest = torch.linspace(x_range[0], x_range[1], n_test) 
            # compute predictive mean and variance
            with torch.no_grad():
                if type(self.gpm) == gp.models.VariationalSparseGP:
                    mean, cov = self.gpm(Xtest, full_cov=True)
                else:
                    mean, cov = self.gpm(Xtest, full_cov=True, noiseless=False)
            sd = cov.diag().sqrt()  # standard deviation at each input point x
            ax.plot(Xtest.numpy(), mean.numpy(), "r", lw=2, label="mean")  # plot the mean
            ax.fill_between(
                Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                (mean - 2.0 * sd).numpy(),
                (mean + 2.0 * sd).numpy(),
                color="C0",
                alpha=0.3,
                label="area in (-2σ, +2σ)"
            )        
        ax.legend()
        plt.savefig("./result_gaussian_process_plot.pdf", bbox_inches='tight')
