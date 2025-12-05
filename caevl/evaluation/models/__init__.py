import os
import torch
from collections.abc import Mapping

from caevl.ft_stage.model.encoder import CaevlFT
from caevl.evaluation.models import mixvpr
from caevl.ae.tools.load_ae import load_ae


def is_state_dict(obj):
    """Check whether obj looks like a PyTorch state_dict."""
    if not isinstance(obj, Mapping):
        return False
    # A valid state_dict has keys that are strings and values that are tensors
    return all(isinstance(k, str) for k in obj.keys()) and any(hasattr(v, "shape") for v in obj.values())


def find_state_dict(obj):
    """Recursively search for a valid state_dict in a nested structure."""
    if is_state_dict(obj):
        return obj

    if isinstance(obj, Mapping):
        for _, v in obj.items():
            result = find_state_dict(v)
            if result is not None:
                return result

    return None


def load_weights(model, weights_path, device, strict=True):
    """
    Load weights into `model` from a file that may store the state_dict
    under various keys, or nested.
    """
    DEVICE = torch.device(device)
    state = torch.load(weights_path, weights_only=False, map_location=DEVICE)

    # If the whole file is directly a state_dict
    if is_state_dict(state):
        model.load_state_dict(state, strict=strict)
        return state

    # Otherwise search inside
    found = find_state_dict(state)
    if found is None:
        raise RuntimeError(
            f"No valid state_dict found in '{weights_path}'. "
        )

    model.load_state_dict(found, strict=strict)
    return found


def get_model(method, backbone=None, descriptors_dimension=None,
              device=None, weights=None):
    if method == "mixvpr":
        model = mixvpr.get_mixvpr(descriptors_dimension=descriptors_dimension)
        if weights is not None:
            load_weights(model, weights, device=device, strict=True)

    elif method == "eigenplaces":
        model = torch.hub.load(
            "gmberton/eigenplaces", "get_trained_model",
            backbone=backbone, fc_output_dim=descriptors_dimension
        )
        if weights is not None:
            load_weights(model, weights, device=device, strict=True)

    elif method == "caevl":
        dir_ = os.path.dirname(weights)
        path_to_config = os.path.join(dir_, 'backbone_config.yml')

        encoder = load_ae(path_to_config)

        # we create a 'false' model as the only thing that matters is the encoder
        model = CaevlFT(backbone=encoder.encoder,
                        projector=None,
                        local_projector=None,
                        device=device,
                        invariance_coeff=0,
                        std_coeff=0,
                        cov_coeff=0,
                        alpha=0)

        load_weights(model, weights, device=device, strict=False)
        model = model.backbone

    return model
