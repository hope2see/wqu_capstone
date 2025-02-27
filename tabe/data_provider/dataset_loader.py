import os
import numpy as np
import pandas as pd
from utils.timefeatures import time_features
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tabe.utils.misc_util import logger

import warnings
warnings.filterwarnings('ignore')


# To be able to keep the temporal sequence of the 'continous onestep test/train' for the test_peroid, 
# validation_set should not be located in the middle of train_set and test_set. 
# In that case, lookup_window (length of seq_len) of the first datapoint of test_set 
# would be overlapped with the validation_set, which should be kept 'unseen' area for validation! 
# So, we moved validation_period to the first part of dataset. 
# [validateion_set (20%) | train_set (50%) | ensemble_train_set (20%) | test_set (10%)]
#
class Dataset_TABE(Dataset):
    def __init__(self, args, root_path, flag='base_train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['val', 'base_train', 'ensemble_train', 'test']
        type_map = {'val': 0, 'base_train': 1, 'ensemble_train': 2, 'test': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_vali = int(len(df_raw) * 0.15) #len(df_raw) - num_base_train - num_test
        assert num_vali > 20+self.seq_len, f"num_vali({num_vali}) should be larger than 20+seq_len({self.seq_len})"
        num_base_train = int(len(df_raw) * 0.5)
        num_ensemble_train = int(len(df_raw) * 0.15)
        num_test = len(df_raw) - num_vali - num_base_train - num_ensemble_train

        logger.info(f"Dataset Period : [Val {num_vali} | Base Train {num_base_train} | Ensemble Train {num_ensemble_train} | Test {num_test}")

        border1s = [0, num_vali - self.seq_len, num_vali + num_base_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_vali, num_vali + num_base_train, num_vali + num_base_train + num_ensemble_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



data_dict = {
    # 'ETTh1': Dataset_ETT_hour,
    # 'ETTh2': Dataset_ETT_hour,
    # 'ETTm1': Dataset_ETT_minute,
    # 'ETTm2': Dataset_ETT_minute,
    # 'custom': Dataset_Custom,
    # 'm4': Dataset_M4,
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
    # 'UEA': UEAloader,
    'TABE': Dataset_TABE,
}


def _data_provider(args, flag, step_by_step=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    shuffle_flag = False if (flag == 'test' or flag == 'TEST' or step_by_step) else True
    drop_last = False
    batch_size = args.batch_size if not step_by_step else 1
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader


# Return (Dataset, DataLoader) tuple for the given parameters. 
# We assume that the values of 'args' are always the same. 
# If a (Dataset, DataLoader) correspoinding to the 'flag' and 'step_by_step' parameters
# has been created before, then the cached objects are retunred. 
_cache_dataset = {}
_cache_dataloader = {}
def get_data_provider(args, flag, step_by_step=False):
    key = flag
    if step_by_step:
        key += '_stb'
    if key not in _cache_dataset:
        _cache_dataset[key], _cache_dataloader[key] = _data_provider(args, flag, step_by_step)
    return _cache_dataset[key], _cache_dataloader[key]
