"""Shared model-loading, PDE-residual, and bootstrap evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model.model import build_model
from model.utils import MinMaxScaler


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    model_arch: str,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["model_config"]
    model = build_model(
        model_arch,
        nk=config["nk"],
        np_size=config["kernel_size"],
        nf=config["hidden_features"],
        input_size=128,
        output_dim=config["output_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_and_residual(
    model,
    u: np.ndarray,
    v: np.ndarray,
    scaler: MinMaxScaler,
    physics_loss,
    device: torch.device,
    batch_size: int,
    pde: str,
):
    predictions = []
    residuals = []
    for start in range(0, len(u), batch_size):
        end = min(start + batch_size, len(u))
        u_batch = torch.from_numpy(u[start:end, None]).to(device)
        v_batch = torch.from_numpy(v[start:end, None]).to(device)
        pred_norm = model(u_batch)
        pred_real = scaler.inverse_transform_tensor(pred_norm)
        predictions.append(pred_norm.cpu().numpy())

        first = pred_real[:, 0, None, None, None]
        second = pred_real[:, 1, None, None, None]
        diffusion = pred_real[:, 2, None, None, None]
        lap_u = physics_loss.laplacian(u_batch)
        lap_v = physics_loss.laplacian(v_batch)
        u2v = u_batch.square() * v_batch
        if pde == "brusselator":
            residual_u = (
                lap_u
                + first
                - (second + 1.0) * u_batch
                + u2v
            )
            residual_v = (
                diffusion * lap_v + second * u_batch - u2v
            )
        elif pde == "schnakenberg":
            residual_u = lap_u + first - u_batch + u2v
            residual_v = diffusion * lap_v + second - u2v
        elif pde == "lengyel_epstein":
            rational = u_batch * v_batch / (
                1.0 + u_batch.square()
            )
            residual_u = (
                lap_u
                + first
                - u_batch
                - 4.0 * rational
                - second
            )
            residual_v = (
                diffusion * lap_v + u_batch - rational + second
            )
        else:
            raise ValueError(f"Unsupported PDE: {pde}")
        mask = (
            u_batch > u_batch.mean(dim=(2, 3), keepdim=True)
        ).to(u_batch.dtype)
        residual_per_sample = (
            ((residual_u.square() + residual_v.square()) * mask)
            .sum(dim=(1, 2, 3))
            / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        )
        residuals.append(residual_per_sample.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(residuals)


def bootstrap_mean_difference(
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
    seed: int,
    samples: int,
) -> dict:
    difference = candidate_values - reference_values
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    chunk_size = 250
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        indices = rng.integers(
            0,
            len(difference),
            size=(count, len(difference)),
        )
        bootstrap_means[start : start + count] = np.mean(
            difference[indices],
            axis=1,
        )
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "mean_physical_minus_pure": float(np.mean(difference)),
        "ci95": [float(low), float(high)],
        "physical_better_probability": float(
            np.mean(bootstrap_means < 0.0)
        ),
    }
