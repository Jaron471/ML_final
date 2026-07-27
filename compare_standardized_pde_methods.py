#!/usr/bin/env python3
"""Compare standardized Pure, Physical, and Soft-Physical PDE experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation_utils import (
    bootstrap_mean_difference,
    load_model,
    predict_and_residual,
)
from model.loss import (
    BrusselatorPhysicsLoss,
    LengyelEpsteinPhysicsLoss,
    SchnakenbergPhysicsLoss,
)
from model.utils import MinMaxScaler
from train_brusselator import calculate_metrics


METHODS = ("pure", "physical", "soft_physical")
PAIRS = (
    ("physical", "pure"),
    ("soft_physical", "pure"),
    ("soft_physical", "physical"),
)
PDE_CONFIG = {
    "schnakenberg": {
        "architecture": "cnn2stride",
        "names": ("a", "b", "d"),
        "physics": SchnakenbergPhysicsLoss,
        "eval_data": Path(
            "schnakenberg_data/schnakenberg_eval_data.npz"
        ),
    },
    "brusselator": {
        "architecture": "cnn1",
        "names": ("A", "B", "d"),
        "physics": BrusselatorPhysicsLoss,
        "eval_data": Path(
            "brusselator_data/brusselator_eval_data.npz"
        ),
    },
    "lengyel_epstein": {
        "architecture": "cnn2stride",
        "names": ("a", "phi", "r"),
        "physics": LengyelEpsteinPhysicsLoss,
        "eval_data": Path(
            "lengyel_epstein_data/lengyel_epstein_eval_data.npz"
        ),
    },
}


def identity_error(
    pde: str,
    params_real: np.ndarray,
    mean_u: np.ndarray,
) -> np.ndarray:
    if pde == "schnakenberg":
        return params_real[:, 0] + params_real[:, 1] - mean_u
    if pde == "brusselator":
        return params_real[:, 0] - mean_u
    if pde == "lengyel_epstein":
        return (
            params_real[:, 0]
            - 5.0 * params_real[:, 1]
            - 5.0 * mean_u
        )
    raise ValueError(f"Unsupported PDE: {pde}")


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def improvement_percent(candidate: float, reference: float) -> float:
    return float(100.0 * (reference - candidate) / reference)


def paired_result(
    candidate: np.ndarray,
    reference: np.ndarray,
    seed: int,
    samples: int,
) -> dict:
    raw = bootstrap_mean_difference(
        candidate,
        reference,
        seed,
        samples,
    )
    return {
        "mean_candidate_minus_reference": raw[
            "mean_physical_minus_pure"
        ],
        "ci95": raw["ci95"],
        "candidate_better_probability": raw[
            "physical_better_probability"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pde",
        choices=tuple(PDE_CONFIG),
        required=True,
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--eval-data", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PDE_CONFIG[args.pde]
    architecture = config["architecture"]
    names = config["names"]
    eval_path = args.eval_data or config["eval_data"]
    device = torch.device(args.device)

    with np.load(eval_path) as data:
        u = data["u"].astype(np.float32)
        v = data["v"].astype(np.float32)
        targets_real = np.stack(
            [data[name] for name in names],
            axis=1,
        ).astype(np.float32)
        ids = data["ids"].astype(np.int32)

    models = {}
    checkpoints = {}
    for method in METHODS:
        stem = f"{args.pde}_{architecture}_{method}_seed{args.seed}"
        models[method], checkpoints[method] = load_model(
            args.experiment_dir / f"{stem}.pt",
            device,
            architecture,
        )

    reference_scaler = checkpoints["pure"]["scaler"]
    for method in METHODS[1:]:
        np.testing.assert_allclose(
            reference_scaler["data_min"],
            checkpoints[method]["scaler"]["data_min"],
        )
        np.testing.assert_allclose(
            reference_scaler["data_max"],
            checkpoints[method]["scaler"]["data_max"],
        )
    for method in METHODS:
        training = checkpoints[method]["training_config"]
        expected = {
            "train_samples": 12000,
            "validation_samples": 4000,
            "eval_samples": 4000,
            "validation_seed": args.seed,
            "selection_metric": "validation_normalized_rmse",
            "hard_projection_at_inference": False,
        }
        for key, value in expected.items():
            if training.get(key) != value:
                raise ValueError(
                    f"{method} has unexpected {key}: "
                    f"{training.get(key)!r} != {value!r}"
                )

    scaler = MinMaxScaler.from_state_dict(reference_scaler)
    scaler.to_device(device)
    targets_norm = scaler.transform(targets_real)
    physics = config["physics"](masked=True).to(device)
    mean_u = u.mean(axis=(1, 2))

    predictions_norm = {}
    predictions_real = {}
    residuals = {}
    identities = {}
    metrics = {}
    for method in METHODS:
        predictions_norm[method], residuals[method] = (
            predict_and_residual(
                models[method],
                u,
                v,
                scaler,
                physics,
                device,
                args.batch_size,
                args.pde,
            )
        )
        predictions_real[method] = scaler.inverse_transform_numpy(
            predictions_norm[method]
        )
        identities[method] = identity_error(
            args.pde,
            predictions_real[method],
            mean_u,
        )
        metrics[method] = calculate_metrics(
            targets_norm,
            predictions_norm[method],
            names,
            scaler,
        )

    true_identity = identity_error(args.pde, targets_real, mean_u)
    pairwise = {}
    for pair_index, (candidate, reference) in enumerate(PAIRS):
        candidate_se = np.square(
            predictions_norm[candidate] - targets_norm
        )
        reference_se = np.square(
            predictions_norm[reference] - targets_norm
        )
        candidate_metrics = metrics[candidate]
        reference_metrics = metrics[reference]
        pair = {
            "candidate": candidate,
            "reference": reference,
            "improvement_percent": {
                "overall_normalized_rmse": improvement_percent(
                    candidate_metrics["normalized_rmse"],
                    reference_metrics["normalized_rmse"],
                ),
                "pde_residual": improvement_percent(
                    float(residuals[candidate].mean()),
                    float(residuals[reference].mean()),
                ),
                "identity_rmse": improvement_percent(
                    rmse(identities[candidate]),
                    rmse(identities[reference]),
                ),
            },
            "paired_bootstrap_normalized_squared_error": {
                "overall": paired_result(
                    candidate_se.mean(axis=1),
                    reference_se.mean(axis=1),
                    args.seed + 1000 * pair_index,
                    args.bootstrap_samples,
                )
            },
            "paired_bootstrap_pde_residual": paired_result(
                residuals[candidate],
                residuals[reference],
                args.seed + 1000 * pair_index + 100,
                args.bootstrap_samples,
            ),
        }
        for parameter_index, name in enumerate(names):
            pair["improvement_percent"][f"{name}_rmse"] = (
                improvement_percent(
                    candidate_metrics["per_parameter"][name][
                        "rmse_real"
                    ],
                    reference_metrics["per_parameter"][name][
                        "rmse_real"
                    ],
                )
            )
            pair["paired_bootstrap_normalized_squared_error"][name] = (
                paired_result(
                    candidate_se[:, parameter_index],
                    reference_se[:, parameter_index],
                    args.seed
                    + 1000 * pair_index
                    + parameter_index
                    + 1,
                    args.bootstrap_samples,
                )
            )
        pairwise[f"{candidate}_vs_{reference}"] = pair

    comparison = {
        "protocol": {
            "pde": args.pde,
            "architecture": architecture,
            "model_seed": args.seed,
            "split_seed": args.seed,
            "train_samples": 12000,
            "validation_samples": 4000,
            "evaluation_samples": len(u),
            "selection_metric": "validation_normalized_rmse",
            "bootstrap_samples": args.bootstrap_samples,
            "evaluation_data": str(eval_path),
        },
        "inference_audit": {
            "prediction_path": "raw model(u) output",
            "hard_projection_at_inference": False,
            "least_squares_at_inference": False,
            "parameter_postprocessing": False,
        },
        "best_epoch": {
            method: checkpoints[method]["training_config"]["best_epoch"]
            for method in METHODS
        },
        "validation": {
            method: checkpoints[method]["validation"]
            for method in METHODS
        },
        "evaluation": metrics,
        "mean_pde_residual": {
            method: float(residuals[method].mean())
            for method in METHODS
        },
        "steady_identity_rmse": {
            "true_parameters": rmse(true_identity),
            **{
                method: rmse(identities[method])
                for method in METHODS
            },
        },
        "pairwise": pairwise,
    }

    result_path = (
        args.experiment_dir
        / f"{args.pde}_{architecture}_three_method_comparison.json"
    )
    result_path.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = (
        args.experiment_dir
        / f"{args.pde}_{architecture}_three_method_comparison.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "best_epoch",
                "overall_normalized_rmse",
                *[f"{name}_rmse_real" for name in names],
                "mean_pde_residual",
                "identity_rmse",
            ]
        )
        for method in METHODS:
            writer.writerow(
                [
                    method,
                    comparison["best_epoch"][method],
                    metrics[method]["normalized_rmse"],
                    *[
                        metrics[method]["per_parameter"][name][
                            "rmse_real"
                        ]
                        for name in names
                    ],
                    comparison["mean_pde_residual"][method],
                    comparison["steady_identity_rmse"][method],
                ]
            )
    np.savez(
        args.experiment_dir
        / f"{args.pde}_{architecture}_three_method_predictions.npz",
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
            f"{method}_pde_residual": residuals[method]
            for method in METHODS
        },
        **{
            f"{method}_identity_error": identities[method]
            for method in METHODS
        },
        true_identity_error=true_identity,
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
