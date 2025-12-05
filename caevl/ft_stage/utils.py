import yaml
import os
from glob import glob

import torch

from caevl.ae.tools.load_ae import load_ae
from caevl.ft_stage.model.projector import Projector


def get_value_in_config(config, key_, default=None):
    """
    Retrieve a value from a nested dictionary configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    key_ : str
        Key to retrieve the value for.
    default : any, optional
        Default value to return if the key is not found, by default None.

    Returns
    -------
    any
        Value corresponding to the key, or default if not found.
    """

    value = config.get(key_)
    if value is not None:
        return value
    for key in config.keys():
        if isinstance(config[key], dict):
            value = get_value_in_config(config[key], key_)
            if value is not None:
                return value
    return default


### AUTOENCODER ###

def load_ae_backbone(config_path, device=None, load_weights=True):
    """
    Construct, and initialize from trained weights, the AutoEncoder backbone from a configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    device : str, optional
        Device to load the model on, by default None.
    load_weights : bool, optional
        Whether to load pre-trained weights, by default True.

    Returns
    -------
    AutoEncoder
        Loaded AutoEncoder model.
    str
        Type of normalization layer used.
    """

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    autoencoder = load_ae(config)
    autoencoder = autoencoder.to(device)
    autoencoder.train()

    dir_, _ = os.path.split(config_path)

    try:
        assert load_weights
        weights = glob(os.path.join(dir_, '*.pth'))[0]
        autoencoder.load_state_dict(torch.load(weights))
        print(f'Loading weights from {weights}')
    except AssertionError:
        pass
    except Exception as e:
        print('Error loading weights', e)
        pass

    norm_layer = 'batch_norm'

    return autoencoder, norm_layer

### LOCAL PROJECTOR ###


def get_local_projector(loader, device, backbone, image_size, norm_layer, nb_neurons=512, inverse_pyramid=False):
    # x = torch.rand(2, 1, *image_size)
    x = next(iter(loader))[0].to(device)
    x_unpooled, _ = backbone.forward_features_unpooled(x)
    b, c, h, w = x_unpooled.shape

    min_diff_to_nb_neurons = nb_neurons
    nb_output_channels = 1
    while nb_output_channels * h * w <= nb_neurons:
        if abs(nb_neurons - nb_output_channels * h * w) <= min_diff_to_nb_neurons:
            min_diff_to_nb_neurons = abs(nb_neurons - nb_output_channels * h * w)
        nb_output_channels += 1

    if not inverse_pyramid:
        local_projector = Projector(features=3 * [nb_output_channels * h * w],
                                    norm_layer=norm_layer)
    else:
        local_projector = Projector(features=[nb_output_channels * h * w // 8,
                                              nb_output_channels * h * w // 16,
                                              nb_output_channels * h * w // 32],
                                    norm_layer=norm_layer)
    return local_projector
