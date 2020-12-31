from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int, hidden_layers: List[int]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (List[int]): List of hidden layers with number of nodes
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        modules = OrderedDict()  # ordered dict of modules that will be passed to a sequential container
        all_layers = [state_size] + hidden_layers + [action_size]
        for index, (in_features, out_features) in enumerate(zip(all_layers[:-1], all_layers[1:])):
            modules['fc' + str(index)] = nn.Linear(in_features, out_features)
            if index < len(hidden_layers):
                modules['relu' + str(index)] = nn.ReLU()

        self.model = nn.Sequential(modules)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)
