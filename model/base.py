from abc import ABC, abstractmethod
import os
import torch
from .model import MLPBase

class Base(ABC):
    """The base class for RL algorithms."""

    def __init__(self, obs_space, action_space, device, save_path, recurrent):
        self.device = device
        self.save_path = save_path
        self.recurrent = recurrent

        self.model = MLPBase(obs_space, action_space).to(self.device)
        
    def save(self, name="acmodel"):
        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch model
        torch.save(self.model, os.path.join(self.save_path, name + filetype))

    @abstractmethod
    def update_policy(self):
        pass