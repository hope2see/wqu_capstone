
import os
import torch.nn as nn
from misc_util import get_config_str

class AbstractModel(object):
    def __init__(self, configs, name):
        self.configs = configs
        self.name = name

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_checkpoint_path(self):
        path = os.path.join(self.configs.checkpoints, get_config_str(self.configs)) 
        path = os.path.join(path, self.name) 
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _get_result_path(self):
        path = os.path.join("./result", get_config_str(self.configs)) 
        path = os.path.join(path, self.name) 
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def load_saved_model(self):
        pass

    # @abstractmethod
    def train(self):
        raise NotImplementedError

    # @abstractmethod
    def test(self):
        raise NotImplementedError

    # @abstractmethod
    def proceed_onestep(self, training: bool = False):
        raise NotImplementedError

