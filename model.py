import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Type, Union


class DxMML(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(DxMML, self).__init__()

        # Initialize the parameters

        # CNN Layer

        # Multimodal layer

    # Define the forward pass
    def forward(self, input):

        return out
