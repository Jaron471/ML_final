#!/usr/bin/env python3
"""Compare GM Pure, Physical, and raw-CNN Soft Physical predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation_utils import bootstrap_mean_difference, load_model
from model.loss import PhysicsLoss
from model.utils import MinMaxScaler


NAMES = ("a", "b", "c", "delta")
METHODS = ("pure", "physical", "soft_physical")


@torch.no_grad()
def predict(
    model,
    u: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    for start in range(0, len(u), batch_size):
        batch = torch.from_numpy(
            u[start : start + batch_size, None]
        ).to(device)
        predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions)


@torch.no_grad()
def residuals_and_constraints(
    u: np.ndarray,
    v: np.ndarray,
    params_real: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physics = PhysicsLoss(dx=1.0, s_diffusion=0.4).to(device)
    frequencies = 2.0 * torch.pi * torch.fft.fftfreq(
        128,
        d=1.0,
        device=device,
    )
    laplace_eigenvalue = -(
        frequencies[:, None].square()
        + frequencies[None, :].square()
    )
    residual_parts = []
    integrated_parts = []
    delta_parts = []
    for start in range(0, len(u), batch_size):
        end = min(start + batch_size, len(u))
        u_batch = torch.from_numpy(
            u[start:end, None]
        ).to(device)
        v_batch = torch.from_numpy(
            v[start:end, None]
        ).to(device)
        params = torch.from_numpy(
            params_real[start:end]
        ).to(device)
        a = params[:, 0, None, None, None]
        b = params[:, 1, None, None, None]
        c = params[:, 2, None, None, None]
        delta = params[:, 3, None, None, None]
        lap_u = physics.laplacian(u_batch)
        lap_v_fd = physics.laplacian(v_batch)
        reaction_u = (
            a
            - b * u_batch
            + u_batch.square()
            / (
                v_batch.clamp_min(1e-6)
                * (1.0 + c * u_batch.square())
            )
        )
        residual_u = 0.4 * lap_u + reaction_u
        residual_v = (
            0.4 * delta * lap_v_fd
            + u_batch.square()
            - v_batch
        )
        mask = (
            u_batch
            > u_batch.mean(dim=(2, 3), keepdim=True)
        ).to(u_batch.dtype)
        residual = (
            (
                residual_u.square()
                + residual_v.square()
            )
            * mask
        ).sum(dim=(1, 2, 3)) / mask.sum(
            dim=(1, 2, 3)
        ).clamp_min(1.0)
        residual_parts.append(residual.cpu().numpy())

        integrated = reaction_u.mean(dim=(1, 2, 3))
        integrated_parts.append(integrated.cpu().numpy())

        lap_v_spectral = torch.fft.ifft2(
            laplace_eigenvalue
            * torch.fft.fft2(v_batch[:, 0])
        ).real
        delta_normal = (
            0.4
            * params[:, 3]
            * lap_v_spectral.square().mean(dim=(1, 2))
            + (
                lap_v_spectral
                * (
                    u_batch[:, 0].square()
                    - v_batch[:, 0]
                )
            ).mean(dim=(1, 2))
        )
        delta_parts.append(delta_normal.cpu().numpy())
    return (
        np.concatenate(residual_parts),
        np.concatenate(integrated_parts),
        np.concatenate(delta_parts),
    )


def improvement_percent(
    candidate: float,
    reference: float,
) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("standardized_split_experiments/gm"),
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=Path("gm_data/gm_eval_data.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("standardized_split_experiments/gm"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics_payloads = {}
    checkpoint_paths = {}
    for method in METHODS:
        stem = f"gm_cnn1_{method}_seed42"
        checkpoint_paths[method] = (
            args.experiment_dir / f"{stem}.pt"
        )
        metrics_payloads[method] = json.loads(
            (
                args.experiment_dir / f"{stem}_metrics.json"
            ).read_text(encoding="utf-8")
        )
    scaler = MinMaxScaler.from_state_dict(
        metrics_payloads["pure"]["scaler"]
    )
    for method in METHODS[1:]:
        np.testing.assert_allclose(
            metrics_payloads[method]["scaler"]["data_min"],
            metrics_payloads["pure"]["scaler"]["data_min"],
        )
        np.testing.assert_allclose(
            metrics_payloads[method]["scaler"]["data_max"],
            metrics_payloads["pure"]["scaler"]["data_max"],
        )
    with np.load(args.eval_data) as data:
        u = data["u"].astype(np.float32)
        v = data["v"].astype(np.float32)
        targets_real = np.stack(
            [data[name] for name in NAMES],
            axis=1,
        ).astype(np.float32)
        ids = data["ids"].astype(np.int32)
    device = torch.device(args.device)
    targets_norm = scaler.transform(targets_real)
    predictions_norm = {}
    for method in METHODS:
        model, checkpoint = load_model(
            checkpoint_paths[method],
            device,
            "cnn1",
        )
        training = checkpoint["training_config"]
        if (
            training.get("hard_projection_at_inference") is not False
            or checkpoint.get("postprocessing")
        ):
            raise ValueError(
                f"{method} checkpoint failed raw-CNN inference audit."
            )
        predictions_norm[method] = predict(
            model,
            u,
            device,
            args.batch_size,
        )
    predictions_real = {
        method: scaler.inverse_transform_numpy(
            predictions_norm[method]
        ).astype(np.float32)
        for method in METHODS
    }
    diagnostics = {}
    true_diagnostics = residuals_and_constraints(
        u,
        v,
        targets_real,
        device,
        args.batch_size,
    )
    for method in METHODS:
        diagnostics[method] = residuals_and_constraints(
            u,
            v,
            predictions_real[method],
            device,
            args.batch_size,
        )

    squared_errors = {
        method: np.square(
            predictions_norm[method] - targets_norm
        )
        for method in METHODS
    }
    comparisons = {}
    pairs = (
        ("physical", "pure"),
        ("soft_physical", "pure"),
        ("soft_physical", "physical"),
    )
    for pair_index, (candidate, reference) in enumerate(pairs):
        comparison = {
            "candidate": candidate,
            "reference": reference,
            "normalized_squared_error": {
                "overall": bootstrap_mean_difference(
                    squared_errors[candidate].mean(axis=1),
                    squared_errors[reference].mean(axis=1),
                    args.seed + 1000 * pair_index,
                    args.bootstrap_samples,
                )
            },
            "original_pde_residual": (
                bootstrap_mean_difference(
                    diagnostics[candidate][0],
                    diagnostics[reference][0],
                    args.seed + 1000 * pair_index + 100,
                    args.bootstrap_samples,
                )
            ),
        }
        for index, name in enumerate(NAMES):
            comparison["normalized_squared_error"][name] = (
                bootstrap_mean_difference(
                    squared_errors[candidate][:, index],
                    squared_errors[reference][:, index],
                    args.seed
                    + 1000 * pair_index
                    + index
                    + 1,
                    args.bootstrap_samples,
                )
            )
        comparisons[f"{candidate}_vs_{reference}"] = comparison

    result = {
        "protocol": {
            "pde": "gierer_meinhardt",
            "architecture": "cnn1",
            "model_seed": args.seed,
            "train_samples": 12000,
            "validation_samples": 4000,
            "evaluation_samples": len(u),
            "selection_metric": "validation_normalized_rmse",
            "bootstrap_samples": args.bootstrap_samples,
            "evaluation_data": str(args.eval_data),
        },
        "inference_audit": {
            "prediction_path": "raw model(u) output",
            "hard_projection_at_inference": False,
            "least_squares_at_inference": False,
            "parameter_postprocessing": False,
        },
        "evaluation": {
            method: metrics_payloads[method]["evaluation"]
            for method in METHODS
        },
        "best_epoch": {
            method: metrics_payloads[method]["training"][
                "best_epoch"
            ]
            for method in METHODS
        },
        "improvement_percent": {},
        "diagnostics": {
            "original_masked_pde_residual_mean": {
                "true_parameters": float(
                    true_diagnostics[0].mean()
                ),
                **{
                    method: float(diagnostics[method][0].mean())
                    for method in METHODS
                },
            },
            "integrated_first_equation_rmse": {
                "true_parameters": float(
                    np.sqrt(np.mean(np.square(true_diagnostics[1])))
                ),
                **{
                    method: float(
                        np.sqrt(
                            np.mean(
                                np.square(diagnostics[method][1])
                            )
                        )
                    )
                    for method in METHODS
                },
            },
            "delta_normal_equation_rmse": {
                "true_parameters": float(
                    np.sqrt(np.mean(np.square(true_diagnostics[2])))
                ),
                **{
                    method: float(
                        np.sqrt(
                            np.mean(
                                np.square(diagnostics[method][2])
                            )
                        )
                    )
                    for method in METHODS
                },
            },
        },
        "pairwise": comparisons,
    }
    for candidate in ("physical", "soft_physical"):
        candidate_metrics = result["evaluation"][candidate]
        pure_metrics = result["evaluation"]["pure"]
        improvements = {
            "overall_normalized_rmse": improvement_percent(
                candidate_metrics["normalized_rmse"],
                pure_metrics["normalized_rmse"],
            )
        }
        for name in NAMES:
            improvements[f"{name}_rmse"] = improvement_percent(
                candidate_metrics["per_parameter"][name][
                    "rmse_real"
                ],
                pure_metrics["per_parameter"][name]["rmse_real"],
            )
        result["improvement_percent"][candidate] = improvements

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = (
        args.output_dir / "gm_cnn1_three_method_comparison.json"
    )
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    with (
        args.output_dir / "gm_cnn1_three_method_comparison.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["metric", "pure", "physical", "soft_physical"]
        )
        writer.writerow(
            [
                "overall_normalized_rmse",
                *[
                    result["evaluation"][method]["normalized_rmse"]
                    for method in METHODS
                ],
            ]
        )
        for name in NAMES:
            writer.writerow(
                [
                    f"{name}_rmse_real",
                    *[
                        result["evaluation"][method]["per_parameter"][
                            name
                        ]["rmse_real"]
                        for method in METHODS
                    ],
                ]
            )
        writer.writerow(
            [
                "original_masked_pde_residual_mean",
                *[
                    result["diagnostics"][
                        "original_masked_pde_residual_mean"
                    ][method]
                    for method in METHODS
                ],
            ]
        )
    np.savez(
        args.output_dir / "gm_cnn1_three_method_predictions.npz",
        ids=ids,
        targets_real=targets_real,
        targets_normalized=targets_norm,
        **{
            f"{method}_predictions_normalized": predictions_norm[
                method
            ]
            for method in METHODS
        },
        **{
            f"{method}_original_pde_residual": diagnostics[
                method
            ][0]
            for method in METHODS
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
