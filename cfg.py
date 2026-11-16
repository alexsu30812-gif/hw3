"""Classifier-free guidance used by our Stable Audio Open sampling loop.

The model predicts v, despite the TA sample's ``noise_pred`` argument names.
The baseline expression follows Diffusers 0.41.0 in the same operation order.
The optional rescaling and schedule are experimental, not assumed improvements.
"""
from __future__ import annotations

import math


def cfg(noise_pred_uncond, noise_pred_cond, guidance_scale):
    """Return u + s(c-u); s=0 is u and s=1 is c (up to floating arithmetic)."""
    scale = float(guidance_scale)
    if not math.isfinite(scale):
        raise ValueError('guidance_scale must be finite')
    if noise_pred_uncond.shape != noise_pred_cond.shape:
        raise ValueError('Conditional and unconditional prediction shapes must match')
    return noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)


def guidance_at_step(scale, index, steps, schedule='constant', end_scale=1.0):
    """A predeclared schedule in denoising-step order, not physical timestep order."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < steps:
        raise ValueError('index is outside the denoising schedule')
    if not all(math.isfinite(float(x)) and float(x) >= 0 for x in (scale, end_scale)):
        raise ValueError('Guidance endpoints must be finite and nonnegative')
    if schedule == 'constant':
        return float(scale)
    if schedule != 'linear':
        raise ValueError('schedule must be constant or linear')
    fraction = 0.0 if steps == 1 else index / (steps - 1)
    return float(scale) + fraction * (float(end_scale) - float(scale))


def guided_prediction(unconditional, conditional, scale, rescale=0.0):
    """Optional per-item standard-deviation rescaling of the guided v prediction.

    r=0 is exactly the baseline. Reductions exclude the batch dimension; the
    reductions use float32. A zero guided standard deviation leaves its scale
    unchanged. This is an audio adaptation of prediction rescaling used in
    diffusion models (Lin et al., https://arxiv.org/abs/2305.08891), not an
    official Stable Audio Open quality guarantee. We apply it to predicted v;
    image-domain findings are not evidence of improved music quality.
    """
    amount = float(rescale)
    if not math.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError('rescale must be in [0,1]')
    guided = cfg(unconditional, conditional, scale)
    if amount == 0:
        return guided
    import torch
    if guided.ndim < 2 or guided[0].numel() < 2:
        raise ValueError('Rescaling needs a batch and at least two values per item')
    dimensions = tuple(range(1, guided.ndim))
    guided32, conditional32 = guided.float(), conditional.float()
    std_guided = guided32.std(dim=dimensions, keepdim=True, correction=0)
    std_conditional = conditional32.std(dim=dimensions, keepdim=True, correction=0)
    epsilon = torch.finfo(torch.float32).eps
    ratio = torch.where(std_guided > epsilon, std_conditional / std_guided.clamp_min(epsilon),
                        torch.ones_like(std_guided))
    return ((1 - amount) * guided32 + amount * guided32 * ratio).to(guided.dtype)
