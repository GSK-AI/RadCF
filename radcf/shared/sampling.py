"""
ODE integration and sampling utilities.

Pure functions for sampling and inversion using Euler/Heun integration.
"""

import torch
from typing import Callable, Dict


def euler_step(x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """Single Euler step: x_next = x + dt * v"""
    return x + dt * v


def integrate_ode(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_start: torch.Tensor,
    t_start: float,
    t_end: float,
    num_steps: int = 50,
    method: str = "euler",
) -> torch.Tensor:
    """
    Integrate ODE from t_start to t_end.

    Flow Matching ODE: dx/dt = v(x, t)
    Integration: x(t_end) = x(t_start) + ∫[t_start→t_end] v(x(τ), τ) dτ

    Args:
        velocity_fn: Callable (x, t) -> v that returns velocity
        x_start: Starting state [B, C, H, W]
        t_start: Starting time (1.0 for sampling, 0.0 for inversion)
        t_end: Ending time (0.0 for sampling, 1.0 for inversion)
        num_steps: Number of integration steps (must be > 0)
        method: "euler" or "heun"

    Returns:
        x_end: Final state [B, C, H, W]

    Raises:
        ValueError: If num_steps <= 0, t_start == t_end, or invalid method
        TypeError: If velocity_fn is not callable

    Note:
        - For sampling (1→0): dt < 0, we integrate backwards
        - For inversion (0→1): dt > 0, we integrate forwards
        - Same formula x = x + dt*v works for both!
    """
    # Validate inputs
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")

    if t_start == t_end:
        raise ValueError(
            f"t_start ({t_start}) must be different from t_end ({t_end}). "
            f"Cannot integrate with zero interval."
        )

    if not callable(velocity_fn):
        raise TypeError(f"velocity_fn must be callable, got {type(velocity_fn)}")

    if method not in ["euler", "heun"]:
        raise ValueError(f"Unknown method: {method}. Must be 'euler' or 'heun'")

    device = x_start.device
    dtype = x_start.dtype

    # Use float64 for numerical stability
    x = x_start

    # Time schedule
    t_steps = torch.linspace(t_start, t_end, num_steps + 1, device=device)

    # Integration loop
    with torch.no_grad():
        for i in range(num_steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            dt = t_next - t_cur  # Automatically has correct sign!

            if method == "heun":
                # Heun's method (2nd order Runge-Kutta)
                v_cur = velocity_fn(x, t_cur)
                x_tmp = euler_step(x, v_cur, dt)
                v_next = velocity_fn(x_tmp, t_next)
                v_avg = 0.5 * (v_cur + v_next)
                x = euler_step(x, v_avg, dt)
            elif method == "euler":
                # Euler's method (1st order)
                v = velocity_fn(x, t_cur)
                x = euler_step(x, v, dt)

    return x.to(dtype)


def euler_sample(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    num_steps: int = 50,
    method: str = "euler",
) -> torch.Tensor:
    """
    Sample from noise to data using ODE integration.

    Convenience function for integrate_ode with t_start=1.0, t_end=0.0.

    Args:
        velocity_fn: Callable (x, t) -> v that returns velocity
        noise: Starting noise [B, C, H, W]
        num_steps: Number of integration steps
        method: "euler" or "heun"

    Returns:
        sample: Generated sample [B, C, H, W]
    """
    return integrate_ode(
        velocity_fn=velocity_fn,
        x_start=noise,
        t_start=1.0,
        t_end=0.0,
        num_steps=num_steps,
        method=method,
    )


def euler_invert(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sample: torch.Tensor,
    num_steps: int = 50,
    method: str = "euler",
) -> torch.Tensor:
    """
    Invert from data to noise using ODE integration.

    Convenience function for integrate_ode with t_start=0.0, t_end=1.0.

    Args:
        velocity_fn: Callable (x, t) -> v that returns velocity
        sample: Data sample [B, C, H, W]
        num_steps: Number of integration steps
        method: "euler" or "heun"

    Returns:
        noise: Inverted noise [B, C, H, W]
    """
    return integrate_ode(
        velocity_fn=velocity_fn,
        x_start=sample,
        t_start=0.0,
        t_end=1.0,
        num_steps=num_steps,
        method=method,
    )


def apply_cfg(
    velocity_fn: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    cond_input,
    uncond_input,
    cfg_scale: float,
) -> torch.Tensor:
    """
    Apply Classifier-Free Guidance.

    Args:
        velocity_fn: Velocity function (x, t, condition) -> v
        x, t: State and time
        cond_input: Conditional input (metadata dict or embeddings)
        uncond_input: Unconditional input (null embeddings)
        cfg_scale: Guidance scale (1.0 = no guidance, must be >= 0)

    Returns:
        v: CFG-weighted velocity

    Raises:
        ValueError: If cfg_scale < 0
    """
    if cfg_scale < 0:
        raise ValueError(f"cfg_scale must be >= 0, got {cfg_scale}")

    if cfg_scale == 1.0:
        return velocity_fn(x, t, cond_input)
    elif cfg_scale == 0.0:
        return velocity_fn(x, t, uncond_input)
    else:
        v_cond = velocity_fn(x, t, cond_input)
        v_uncond = velocity_fn(x, t, uncond_input)
        return v_uncond + cfg_scale * (v_cond - v_uncond)
