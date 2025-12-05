import torch
import torch.nn as nn


class Projector(torch.nn.Module):
    """
    A neural network projector module that projects input features through a series of linear layers
    with optional normalization and ReLU activations.

    Parameters
    ----------
    features : list of int, optional
        The number of features in each layer of the projector. The default is [8192, 8192, 8192].
    norm_layer : str, optional
        The type of normalization layer to use. Options are 'batch_norm' for BatchNorm1d and 'layer_norm' for LayerNorm.
        The default is 'batch_norm'.
    """

    def __init__(self, features=[8192, 8192, 8192], norm_layer='batch_norm'):
        super(Projector, self).__init__()

        self.norm_layer = norm_layer

        layers = [
            nn.LazyLinear(out_features=features[0]),
            nn.BatchNorm1d(features[0]) if self.norm_layer == 'batch_norm' else nn.LayerNorm(features[0]),
            nn.ReLU()
        ]

        for i in range(1, len(features) - 1):
            layers.append(nn.Linear(in_features=features[i - 1], out_features=features[i]))
            layers.append(nn.BatchNorm1d(features[i]) if self.norm_layer == 'batch_norm' else nn.LayerNorm(features[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=features[-2], out_features=features[-1]))

        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        out = self.projector(x)
        return out
