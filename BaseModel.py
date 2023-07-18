import os
import utils
import torch
from torch import nn
from pathlib import Path
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.res_dir = os.path.join(utils.get_res_path(), self.TRAINED_MODELS_DIR)
        self.device = utils.get_device()

    @property
    @abstractmethod
    def MODEL_NAME(self):
        ...

    @property
    @abstractmethod
    def TRAINED_MODELS_DIR(self):
        ...

    def save_model(self, res_dir=None, model_name=None):
        model_name = self.MODEL_NAME if model_name is None else model_name
        res_dir = self.res_dir if res_dir is None else res_dir
        Path(res_dir).mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), os.path.join(res_dir, model_name))

    def load_model(self, res_dir=None, model_name=None):
        model_name = self.MODEL_NAME if model_name is None else model_name
        res_dir = self.res_dir if res_dir is None else res_dir
        print(f'Loading model {model_name}...')
        self.load_state_dict(torch.load(os.path.join(res_dir, model_name), map_location=self.device))
