"""Reference utilities for implementing FASTER in a SmolVLA-style policy.

This file is intentionally written as an implementation note plus reusable helper code.
It is not currently imported by SmolVLA at runtime.

The FASTER algorithm changes three parts of a flow-matching VLA stack:

1. Dataloader
   - sample a control delay `d`
   - build a prefix mask `m_i = 1(i >= d)`

2. Loss construction
   - sample a mixed schedule gate `z ~ Bernoulli(p)`
   - sample a global time `rho ~ Uniform(0, 1)`
   - convert `rho` into token-wise local times `tau`
   - train with a masked flow-matching loss so prefix actions do not contribute

3. Inference loop
   - replace scalar time integration with vector time integration
   - dispatch actions when their local time first reaches zero
   - stop early once the active execution window is fully denoised

The model architecture does not need to change, but the time embedding path does.
Any model using this logic must support token-wise time tensors of shape `(B, H)`.

Paper-aligned defaults:
- prediction horizon H = 50
- denoising steps N = 10
- first-action hit time u_d = (N - 1) / N = 0.9
- HAS exponent alpha = 0.6
- mixed-schedule probability p = 0.5
- max simulated delay d_max = 10
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FASTERConfig:
    """Hyperparameters used by the reference FASTER implementation."""

    prediction_horizon: int = 50
    num_steps: int = 10
    first_action_hit_time: float = 0.9
    alpha: float = 0.6
    mixed_schedule_probability: float = 0.5
    max_delay: int = 10


@dataclass(frozen=True)
class FASTERTrainingBatch:
    """Outputs required to build a FASTER-style training example.

    Shapes:
    - delays: `(B,)`
    - mask: `(B, H)`
    - use_has_schedule: scalar bool stored as a rank-0 tensor
    - global_time: `(B,)`
    - local_time: `(B, H)`
    """

    delays: Tensor
    mask: Tensor
    use_has_schedule: Tensor
    global_time: Tensor
    local_time: Tensor


@dataclass(frozen=True)
class FASTERStepSchedule:
    """Per-step inference schedule for vectorized Euler integration.

    Shapes:
    - rho: scalar tensor
    - rho_next: scalar tensor
    - tau: `(H,)`
    - tau_next: `(H,)`
    - delta_tau: `(1, H, 1)` for broadcasting with `(B, H, A)`
    """

    rho: Tensor
    rho_next: Tensor
    tau: Tensor
    tau_next: Tensor
    delta_tau: Tensor


def sample_delays_and_mask(
    batch_size: int,
    horizon: int,
    max_delay: int,
    *,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Sample per-sample delays and build the binary action mask.

    Delay sampling follows the paper-style simulation of asynchronous control:

    `d ~ Uniform({0, ..., d_max})`

    The mask is then:

    `m_i = 1(i >= d)`

    Prefix actions where `i < d` remain in the input chunk as conditioning context but should
    be excluded from the loss.
    """

    delays = torch.randint(0, max_delay + 1, (batch_size,), device=device)
    token_ids = torch.arange(horizon, device=device).unsqueeze(0)
    mask = token_ids >= delays.unsqueeze(1)
    return delays, mask


def sample_mixed_schedule(
    batch_size: int,
    probability: float,
    *,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Sample the mixed-schedule gate and the global time.

    The paper describes one Bernoulli draw per batch:

    `z ~ Bernoulli(p)`
    `rho ~ Uniform(0, 1)`

    We return:
    - `use_has_schedule`: scalar bool tensor
    - `global_time`: `(B,)` tensor so each sample can still carry its own scalar rho if desired
    """

    use_has_schedule = torch.rand((), device=device) < probability
    global_time = torch.rand(batch_size, device=device)
    return use_has_schedule, global_time


def compute_hit_times(
    delays: Tensor,
    horizon: int,
    alpha: float,
    first_action_hit_time: float,
) -> Tensor:
    """Compute the per-token hit-time vector `u`.

    For every valid token position `i >= d`:

    `u_i = (1 - (i - d) / max(H - 1 - d, 1))^alpha * u_d`

    Tokens in the prefix region `i < d` are left at zero because they are already treated as
    completed conditioning tokens.
    """

    token_ids = torch.arange(horizon, device=delays.device).unsqueeze(0)
    delay_matrix = delays.unsqueeze(1)
    valid = token_ids >= delay_matrix
    denom = torch.clamp(horizon - 1 - delay_matrix, min=1)
    normalized = (token_ids - delay_matrix).to(torch.float32) / denom.to(torch.float32)
    remaining = torch.clamp(1.0 - normalized, min=0.0)

    hit_times = torch.zeros_like(remaining)
    hit_times[valid] = remaining[valid].pow(alpha) * first_action_hit_time
    return hit_times


def compute_training_local_time(
    delays: Tensor,
    global_time: Tensor,
    horizon: int,
    use_has_schedule: bool | Tensor,
    alpha: float,
    first_action_hit_time: float,
) -> Tensor:
    """Build the local-time tensor used during training.

    The mixed schedule has two cases.

    HAS case:
    `tau_i = max(0, (rho - u_i) / (1 - u_i))`

    Baseline case:
    `tau_i = rho`

    In both cases, prefix tokens with `i < d` are forced to zero.
    """

    hit_times = compute_hit_times(delays, horizon, alpha, first_action_hit_time)
    valid = torch.arange(horizon, device=delays.device).unsqueeze(0) >= delays.unsqueeze(1)
    tau = torch.zeros_like(hit_times)

    if bool(torch.as_tensor(use_has_schedule)):
        denom = torch.clamp(1.0 - hit_times, min=1e-6)
        tau[valid] = torch.clamp((global_time.unsqueeze(1) - hit_times)[valid] / denom[valid], min=0.0)
    else:
        tau[valid] = global_time.unsqueeze(1).expand_as(tau)[valid]

    return tau


def build_training_batch(
    batch_size: int,
    config: FASTERConfig,
    *,
    device: torch.device | str,
) -> FASTERTrainingBatch:
    """Sample all schedule tensors needed by the training path."""

    delays, mask = sample_delays_and_mask(
        batch_size=batch_size,
        horizon=config.prediction_horizon,
        max_delay=config.max_delay,
        device=device,
    )
    use_has_schedule, global_time = sample_mixed_schedule(
        batch_size=batch_size,
        probability=config.mixed_schedule_probability,
        device=device,
    )
    local_time = compute_training_local_time(
        delays=delays,
        global_time=global_time,
        horizon=config.prediction_horizon,
        use_has_schedule=use_has_schedule,
        alpha=config.alpha,
        first_action_hit_time=config.first_action_hit_time,
    )
    return FASTERTrainingBatch(
        delays=delays,
        mask=mask,
        use_has_schedule=torch.as_tensor(use_has_schedule),
        global_time=global_time,
        local_time=local_time,
    )


def mix_actions_with_noise(actions: Tensor, noise: Tensor, local_time: Tensor) -> Tensor:
    """Construct the noisy action input `A_t^tau`.

    `A_t^tau = tau * epsilon + (1 - tau) * A_t`

    Expected shapes:
    - actions: `(B, H, A)`
    - noise: `(B, H, A)`
    - local_time: `(B, H)`
    """

    tau = local_time.unsqueeze(-1)
    return tau * noise + (1.0 - tau) * actions


def masked_flow_matching_loss(pred_velocity: Tensor, actions: Tensor, noise: Tensor, mask: Tensor) -> Tensor:
    """Compute the masked FASTER training loss.

    Velocity target:
    `u_t = epsilon - A_t`

    Paper-style masking:
    - remove prefix actions from the objective
    - normalize by the number of valid action positions

    This helper first averages over the action dimension and then masks over token positions.
    That keeps the normalization stable when the action dimension changes.
    """

    target_velocity = noise - actions
    per_token_mse = (pred_velocity - target_velocity).pow(2).mean(dim=-1)
    valid = mask.to(dtype=per_token_mse.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return ((per_token_mse * valid).sum(dim=1) / denom).mean()


def build_inference_step_schedule(
    step: int,
    num_steps: int,
    delay: int,
    horizon: int,
    alpha: float,
    first_action_hit_time: float,
    *,
    device: torch.device | str,
) -> FASTERStepSchedule:
    """Build one FASTER inference step.

    Reverse-time schedule:
    - `rho^j = (N - j) / N`
    - `rho^(j+1) = (N - j - 1) / N`

    Local times are then computed from the hit-time vector, and the Euler step becomes token-wise:
    - `Delta tau = tau^(j+1) - tau^j`
    """

    rho = torch.tensor((num_steps - step) / num_steps, device=device, dtype=torch.float32)
    rho_next = torch.tensor((num_steps - step - 1) / num_steps, device=device, dtype=torch.float32)

    delays = torch.tensor([delay], device=device)
    hit_times = compute_hit_times(
        delays=delays,
        horizon=horizon,
        alpha=alpha,
        first_action_hit_time=first_action_hit_time,
    ).squeeze(0)
    valid = torch.arange(horizon, device=device) >= delay
    denom = torch.clamp(1.0 - hit_times, min=1e-6)

    tau = torch.zeros(horizon, device=device)
    tau_next = torch.zeros(horizon, device=device)
    tau[valid] = torch.clamp((rho - hit_times)[valid] / denom[valid], min=0.0)
    tau_next[valid] = torch.clamp((rho_next - hit_times)[valid] / denom[valid], min=0.0)

    delta_tau = (tau_next - tau).view(1, horizon, 1)
    return FASTERStepSchedule(
        rho=rho,
        rho_next=rho_next,
        tau=tau,
        tau_next=tau_next,
        delta_tau=delta_tau,
    )


def dispatch_mask(previous_tau: Tensor, next_tau: Tensor) -> Tensor:
    """Return the tokens that have just become ready for dispatch.

    A token is newly dispatchable if it was still denoising before the step and becomes complete
    at the end of the step.
    """

    return (previous_tau > 0) & (next_tau <= 0)


def should_stop_early(next_tau: Tensor, delay: int, execution_horizon: int) -> bool:
    """Check the early-stopping condition for the active execution window.

    Stop once all actions in `[d, d + s - 1]` have reached `tau = 0`.
    """

    end = min(delay + execution_horizon, next_tau.shape[0])
    if end <= delay:
        return True
    return bool(torch.all(next_tau[delay:end] <= 0))


def build_model_timestep_input(local_time: Tensor, batch_size: int) -> Tensor:
    """Convert a `(H,)` local-time vector into the `(B, H)` tensor expected by the model.

    This mirrors the tensor that a FASTER-style scheduler would pass to the denoiser on each step.
    """

    return local_time.unsqueeze(0).expand(batch_size, -1)


__all__ = [
    "FASTERConfig",
    "FASTERTrainingBatch",
    "FASTERStepSchedule",
    "build_inference_step_schedule",
    "build_model_timestep_input",
    "build_training_batch",
    "compute_hit_times",
    "compute_training_local_time",
    "dispatch_mask",
    "masked_flow_matching_loss",
    "mix_actions_with_noise",
    "sample_delays_and_mask",
    "sample_mixed_schedule",
    "should_stop_early",
]
