"""
Based on https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py
"""

from typing import Any, Dict

import numpy as np

from .gaussian_diffusion import (
    GaussianDiffusion,
    SpacedDiffusion,
    get_named_beta_schedule,
    space_timesteps,
)

BASE_DIFFUSION_CONFIG = {
    "channel_biases": [0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
    "channel_scales": [2.0, 2.0, 2.0, 0.007843137255, 0.007843137255, 0.007843137255],
    "mean_type": "epsilon",
    "schedule": "cosine",
    "timesteps": 1024,
}

DIFFUSION_CONFIGS = {
    "base40M-imagevec": BASE_DIFFUSION_CONFIG,
    "base40M-textvec": BASE_DIFFUSION_CONFIG,
    "base40M-uncond": BASE_DIFFUSION_CONFIG,
    "base40M": BASE_DIFFUSION_CONFIG,
    "base300M": BASE_DIFFUSION_CONFIG,
    "base1B": BASE_DIFFUSION_CONFIG,
    "upsample": {
        "channel_biases": [0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        "channel_scales": [2.0, 2.0, 2.0, 0.007843137255, 0.007843137255, 0.007843137255],
        "mean_type": "epsilon",
        "schedule": "linear",
        "timesteps": 1024,
    },
}


def diffusion_from_config(config: Dict[str, Any]) -> GaussianDiffusion:
    schedule = config["schedule"]
    steps = config["timesteps"]
    respace = config.get("respacing", None)
    mean_type = config.get("mean_type", "epsilon")
    var_type = config.get("var_type", "learned_range")
    loss_type = config.get("loss_type", "mse")
    offset_noise = config.get("offset_noise", False)

    betas = get_named_beta_schedule(schedule, steps)
    channel_scales = config.get("channel_scales", None)
    channel_biases = config.get("channel_biases", None)
    if channel_scales is not None:
        channel_scales = np.array(channel_scales)
    if channel_biases is not None:
        channel_biases = np.array(channel_biases)
    kwargs = dict(
        betas=betas,
        model_mean_type=mean_type,
        model_var_type=var_type,
        loss_type=loss_type,
        channel_scales=channel_scales,
        channel_biases=channel_biases,
        offset_noise=offset_noise,
    )
    if respace is None:
        return GaussianDiffusion(**kwargs)
    else:
        return SpacedDiffusion(use_timesteps=space_timesteps(steps, respace), **kwargs)