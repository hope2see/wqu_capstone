
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def plot_multiple_trials(trials_list, trial_labels, value_key, title=None, filepath=None):
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
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')


def plot_forecast(y, y_hat, title=None, filepath=None):
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame({
        'Ground Truth': y,
        'Forecast': y_hat
    })
    df.plot()
    if title:
        plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()    
    plt.grid(True)
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')


def plot_forecast_result(truth, adjuster_pred, combiner_pred, base_preds, basemodels, filepath=None):
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
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')



def plot_weights(weights_hist, title=None, filepath=None):
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
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')


def plot_hpo_result(HP, trials, title, filepath=None):
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

    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')



def plot_gpmodel(gpm, plot_observed_data=True, plot_predictions=True, n_test=500, x_range=None, filepath=None):
    if x_range is None:
        min = gpm.X.numpy().min()
        max = gpm.X.numpy().max()
        x_range = (min - abs(max-min)*0.1, max + abs(max-min)*0.1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Prediction Deviation Distribution (Analyzed in Gaussian Process)")
    ax.set_ylabel("Prediction deviation at day t")
    ax.set_xlabel("Prediction deviation at day t-1")

    if plot_observed_data:
        ax.plot(gpm.X.numpy(), gpm.y.numpy(), "kx", label="observations")

    if plot_predictions:
        Xtest = torch.linspace(x_range[0], x_range[1], n_test) 
        # compute predictive mean and variance
        with torch.no_grad():
            if type(gpm) == gp.models.VariationalSparseGP:
                mean, cov = gpm(Xtest, full_cov=True)
            else:
                mean, cov = gpm(Xtest, full_cov=True, noiseless=False)
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

    # result_path = self._get_result_path()
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')



def report_losses(y, y_hat_adj, y_hat_cbm, y_hat_bsm, filepath=None):
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
