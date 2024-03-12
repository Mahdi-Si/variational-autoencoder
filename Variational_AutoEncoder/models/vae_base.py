from typing import List, Any, Tuple
import torch
from torch import nn
import numpy as np

from abc import ABC, abstractmethod

class BaseVAE(nn.Module):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__(BaseVAE, self)

    @abstractmethod
    def encode(self, input: torch.tensor):
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent_z: torch.tensor):
        raise NotImplementedError