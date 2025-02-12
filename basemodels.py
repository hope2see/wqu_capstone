# import sys
# # Add Time-Series-Library directory to module lookup paths
# sys.path.append('Time-Series-Library')

import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

from dataset_loader import data_provider

warnings.filterwarnings('ignore')

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX



class BaseModel(object):
    def __init__(self, configs, name):
        self.configs = configs
        self.name = name

    # @abstractmethod
    def train(self):
        raise NotImplementedError

    # @abstractmethod
    def test(self, save_dir):
        raise NotImplementedError

    # @abstractmethod
    def proceed_onestep(self, training: bool = False):
        raise NotImplementedError

    def load_saved_model(self, setting_str=None):
        pass


# NOTE 
# Statistical models (ETS, and SARIMA) are available only for univariate forecasting,
# So, they are fitted only using the target variable.
class StatisticalModel(BaseModel):
    def __init__(self, configs, name):
        super().__init__(configs, name) 

    def _fit(self, endog):
        raise NotImplementedError

    # Statistical Models do not need to be trained, 
    # since they can fit with the 'seq_len' number of data points when prediction is needed.
    def train(self, save_dir):
        pass

    def load_saved_model(self, setting_str=None):
        pass

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1
        endog = batch_y[0, :self.configs.seq_len, -1] # shape=(B,S+1,F) B(Batch Size)=1, S(Sequence Length)+1, F(Feature Dimension)
        endog = endog.numpy()
        pred = self._fit(endog).forecast(steps=1)
        truth = batch_y[0, -1, -1] 
        loss = criterion(torch.tensor(pred), truth).item()
        return loss, pred[0]
    

class EtsModel(StatisticalModel):
    def __init__(self, configs):
        super().__init__(configs, "ETS") 

    def _fit(self, endog):
        # NOTE: Use auto-finding of the hyperparameters for ETS
        return ExponentialSmoothing(endog, trend='add', damped_trend=True).fit()


class SarimaModel(StatisticalModel):
    def __init__(self, configs):
        super().__init__(configs, "SARIMA") 

    def _fit(self, endog_y):
        # NOTE: Use auto-finding of the hyperparameters for SARIMA
        return SARIMAX(endog_y, order=(1,1,0), trend='ct', enforce_stationarity=False).fit(disp=False)
    


# Written by referencing Time-Series-Library/exp_longterm_forecasting.py,exp_basic.py
class NeurlNetModel(BaseModel):
    def __init__(self, configs, name, model):
        super().__init__(configs, name)
        self.device = self._acquire_device()
        self.model = self._build_model(model).to(self.device)

    def _build_model(self, model):
        model = model.Model(self.configs).float()
        if self.configs.use_multi_gpu and self.configs.use_gpu: # CHECK
            model = nn.DataParallel(model, device_ids=self.configs.device_ids)
        return model

    def _acquire_device(self):
        if self.configs.use_gpu and self.configs.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.configs.gpu) if not self.configs.use_multi_gpu else self.configs.devices
            device = torch.device('cuda:{}'.format(self.configs.gpu))
            print('Use GPU: cuda:{}'.format(self.configs.gpu))
        elif self.configs.use_gpu and self.configs.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.configs, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def _forward_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.configs.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.configs.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.configs.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.configs.features == 'MS' else 0
        outputs = outputs[:, -self.configs.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.configs.pred_len:, f_dim:].to(self.device)
        loss = criterion(outputs, batch_y)
        return loss, outputs



    # TODO : NOTE:  
    # When training is needed, only one epoch (for the given batch) is performed! 
    # And, validation is not performed.. 
    # Is it okay?? 
    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        # forward onestep, and get the prediction and loss
        self.model.eval()
        loss, outputs = self._forward_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion)
        loss = loss.item()
        pred = outputs[0, -1, -1].item()

        if training: 
            self._train_batch_with_validation(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion)

        return loss, pred


    def load_saved_model(self, setting_str):
        print('loading model')
        path = os.path.join(self.configs.checkpoints, setting_str) + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(path))

    # NOTE 
    # Is it okay to train repeatitively with the same one batch?? 
    # Or, do we have to use all 'train period' datapoint? 
    def _train_batch_with_validation(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion):
        vali_data, vali_loader = self._get_data(flag='val')

        # path = os.path.join(self.configs.checkpoints, setting_str)
        path = os.path.join(self.configs.checkpoints, "temp_checkpoints")
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = 1 # len(train_loader)
        early_stopping = EarlyStopping(patience=self.configs.patience, verbose=True)

        model_optim = self._select_optimizer()
        # criterion = self._select_criterion()

        if self.configs.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.configs.train_epochs):
            train_loss = []
            self.model.train()
            model_optim.zero_grad()

            loss, _ = self._forward_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion)
            train_loss.append(loss.item())

            if self.configs.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self._validate(vali_data, vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.configs)

        print(f"_train_batch_with_validation: cost time: {time.time() - time_now}")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))


    def train(self, setting_str):
        train_data, train_loader = self._get_data(flag='base_train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.configs.checkpoints, setting_str)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.configs.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.configs.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.configs.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                loss, _ = self._forward_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.configs.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.configs.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self._validate(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.configs)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
    

    # NOTE: 
    # The logic of this function is subordinate to train() function, 
    # and not intended to be used independently.
    def _validate(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.configs.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.configs.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.configs.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.configs.features == 'MS' else 0
                outputs = outputs[:, -self.configs.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.configs.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train() 
        return total_loss


    # NOTE
    # 'vali_data' is just used for evaluating the loss of the trained parameters.
    # So, test() can be called without 'vali_data' being used for train the model. 
    # Given the fact that test_data is actually those data that comes after vali_data, 
    # is it okay to exclude vali_data for training the model?
    def test(self, setting, load_saved_model=False):
        if load_saved_model:
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.load_saved_model(setting)

        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.configs.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.configs.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.configs.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.configs.features == 'MS' else 0
                outputs = outputs[:, -self.configs.pred_len:, :]
                batch_y = batch_y[:, -self.configs.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.configs.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.configs.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.configs.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return preds




