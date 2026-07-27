#!/usr/bin/env python3
"""Train paired three-parameter reaction-diffusion inverse experiments.

This entry point intentionally runs one loss configuration at a time so two
otherwise identical runs can be placed on separate GPUs:

    CUDA_VISIBLE_DEVICES=1 python train_brusselator.py --loss-type pure
    CUDA_VISIBLE_DEVICES=2 python train_brusselator.py --loss-type physical

The reproducible three-way protocol reserves a fixed validation subset from
the 16,000-sample training file.  It uses validation normalized RMSE to select
the checkpoint and evaluates that checkpoint once on the separate 4,000-sample
file.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

# Required by deterministic CUDA matrix multiplications.  It must be set
# before the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from model.dataset import TuringDataset
from model.loss import (
    BrusselatorPhysicsLoss,
    LengyelEpsteinPhysicsLoss,
    SchnakenbergPhysicsLoss,
)
from model.model import MODEL_ARCHES, build_model
from model.soft_constraints import SoftPDEConstraintLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def learning_rate_at_epoch(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    warmup_epochs = max(1, round(0.1 * total_epochs))
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, cosine_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def physics_weight(
    loss_type: str,
    epoch: int,
    maximum: float,
    start_epoch: int,
) -> float:
    if loss_type == "pure" or epoch < start_epoch:
        return 0.0
    return maximum


def soft_constraint_weight(
    loss_type: str,
    epoch: int,
    maximum: float,
    start_epoch: int,
    ramp_epochs: int,
) -> float:
    """Linearly introduce a training-only soft constraint."""
    if (
        loss_type != "soft_physical"
        or maximum <= 0.0
        or epoch < start_epoch
    ):
        return 0.0
    progress = (epoch - start_epoch + 1) / max(1, ramp_epochs)
    return maximum * min(1.0, progress)


def calculate_metrics(
    targets_norm: np.ndarray,
    predictions_norm: np.ndarray,
    parameter_names: tuple[str, ...],
    scaler,
) -> dict:
    targets_real = scaler.inverse_transform_numpy(targets_norm)
    predictions_real = scaler.inverse_transform_numpy(predictions_norm)
    error_norm = predictions_norm - targets_norm
    error_real = predictions_real - targets_real

    rmse_norm = np.sqrt(np.mean(error_norm**2, axis=0))
    mae_norm = np.mean(np.abs(error_norm), axis=0)
    rmse_real = np.sqrt(np.mean(error_real**2, axis=0))
    mae_real = np.mean(np.abs(error_real), axis=0)
    parameter_ranges = scaler.range_val
    range_nrmse = rmse_real / parameter_ranges
    ss_res = np.sum(error_real**2, axis=0)
    ss_tot = np.sum(
        (targets_real - np.mean(targets_real, axis=0)) ** 2,
        axis=0,
    )
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    per_parameter = {}
    for index, name in enumerate(parameter_names):
        per_parameter[name] = {
            "rmse_real": float(rmse_real[index]),
            "mae_real": float(mae_real[index]),
            "rmse_normalized": float(rmse_norm[index]),
            "mae_normalized": float(mae_norm[index]),
            "range_nrmse": float(range_nrmse[index]),
            "r2": float(r2[index]),
        }

    return {
        "normalized_rmse": float(np.sqrt(np.mean(error_norm**2))),
        "normalized_mae": float(np.mean(np.abs(error_norm))),
        "mean_range_nrmse": float(np.mean(range_nrmse)),
        "mean_r2": float(np.mean(r2)),
        "per_parameter": per_parameter,
    }


@torch.no_grad()
def evaluate(model, loader, device, parameter_names, scaler) -> dict:
    model.eval()
    all_targets = []
    all_predictions = []
    for u_batch, _, params_target in loader:
        predictions = model(
            u_batch.to(device, non_blocking=True)
        )
        all_targets.append(params_target.numpy())
        all_predictions.append(predictions.cpu().numpy())
    return calculate_metrics(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        parameter_names,
        scaler,
    )


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    pde_specs = {
        "brusselator": {
            "parameter_names": ("A", "B", "d"),
            "physics_loss": BrusselatorPhysicsLoss,
        },
        "schnakenberg": {
            "parameter_names": ("a", "b", "d"),
            "physics_loss": SchnakenbergPhysicsLoss,
        },
        "lengyel_epstein": {
            "parameter_names": ("a", "phi", "r"),
            "physics_loss": LengyelEpsteinPhysicsLoss,
        },
    }
    spec = pde_specs[args.pde]
    parameter_names = spec["parameter_names"]
    supervised_weights = torch.tensor(
        args.supervised_weights,
        dtype=torch.float32,
        device=device,
    )
    supervised_weights = (
        supervised_weights / supervised_weights.mean()
    )
    soft_feature_gradient_scale = (
        0.0
        if args.soft_head_only
        else args.soft_feature_gradient_scale
    )

    validation_set = None
    validation_indices = None
    if args.validation_count:
        with np.load(args.train_data) as data:
            sample_count = len(data["u"])
        if args.validation_count >= sample_count:
            raise ValueError(
                "--validation-count must be smaller than the training set."
            )
        permutation = np.random.default_rng(
            args.validation_seed
        ).permutation(sample_count)
        validation_indices = permutation[: args.validation_count]
        development_indices = permutation[args.validation_count :]
        train_set = TuringDataset(
            args.train_data,
            parameter_names=parameter_names,
            indices=development_indices,
        )
        validation_set = TuringDataset(
            args.train_data,
            scaler=train_set.scaler,
            parameter_names=parameter_names,
            indices=validation_indices,
        )
        eval_set = TuringDataset(
            args.eval_data,
            scaler=train_set.scaler,
            parameter_names=parameter_names,
        )
    else:
        development_indices = None
        train_set = TuringDataset(
            args.train_data,
            parameter_names=parameter_names,
        )
        eval_set = TuringDataset(
            args.eval_data,
            scaler=train_set.scaler,
            parameter_names=parameter_names,
        )
    if tuple(train_set.parameter_names) != parameter_names:
        raise ValueError(
            f"Expected {args.pde} parameters {parameter_names}."
        )
    train_set.scaler.to_device(device)

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_set,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if validation_set is not None
        else None
    )

    model = build_model(
        args.model_arch,
        nk=args.nk,
        np_size=args.kernel_size,
        nf=args.hidden_features,
        input_size=128,
        output_dim=3,
    ).to(device)
    if (
        soft_feature_gradient_scale < 1.0
        and not hasattr(model, "fc_out")
    ):
        raise ValueError(
            "Partial soft-feature gradients require a CNN model with "
            "an fc_out head."
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    physics_loss_fn = spec["physics_loss"](
        grid_size=128,
        dx=1.0,
        masked=not args.no_physics_mask,
    ).to(device)
    soft_constraint_loss_fn = SoftPDEConstraintLoss(
        args.pde,
        grid_size=128,
        dx=1.0,
    ).to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{args.pde}_{args.model_arch}_{args.loss_type}"
        f"_seed{args.seed}"
    )
    checkpoint_path = args.output_dir / f"{run_name}.pt"
    metrics_path = args.output_dir / f"{run_name}_metrics.json"
    history_path = args.output_dir / f"{run_name}_history.json"
    split_path = (
        args.output_dir
        / f"{args.pde}_validation_split_seed{args.validation_seed}.npz"
        if validation_set is not None
        else None
    )
    if split_path is not None:
        if split_path.exists():
            with np.load(split_path) as split:
                np.testing.assert_array_equal(
                    split["train_indices"],
                    development_indices,
                )
                np.testing.assert_array_equal(
                    split["validation_indices"],
                    validation_indices,
                )
        else:
            np.savez(
                split_path,
                train_indices=development_indices,
                validation_indices=validation_indices,
                split_seed=np.int64(args.validation_seed),
                source_samples=np.int64(sample_count),
            )
    history = []
    best_epoch = None
    best_validation_rmse = float("inf")
    best_state = None
    started_at = time.time()

    print(
        json.dumps(
            {
                "run": run_name,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "train_samples": len(train_set),
                "validation_samples": (
                    len(validation_set)
                    if validation_set is not None
                    else 0
                ),
                "eval_samples": len(eval_set),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lambda_physics": (
                    args.lambda_physics
                    if args.loss_type == "physical"
                    or args.loss_type == "soft_physical"
                    else 0.0
                ),
                "lambda_identity": (
                    args.lambda_identity
                    if args.loss_type == "soft_physical"
                    else 0.0
                ),
                "lambda_elimination": (
                    args.lambda_elimination
                    if args.loss_type == "soft_physical"
                    else 0.0
                ),
                "supervised_weights": supervised_weights.tolist(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        lr = learning_rate_at_epoch(
            epoch,
            args.epochs,
            args.learning_rate,
            args.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        lambda_physics = physics_weight(
            args.loss_type,
            epoch,
            args.lambda_physics,
            args.physics_start_epoch,
        )
        lambda_identity = soft_constraint_weight(
            args.loss_type,
            epoch,
            args.lambda_identity,
            args.soft_constraint_start_epoch,
            args.soft_constraint_ramp_epochs,
        )
        lambda_elimination = soft_constraint_weight(
            args.loss_type,
            epoch,
            args.lambda_elimination,
            args.soft_constraint_start_epoch,
            args.soft_constraint_ramp_epochs,
        )

        supervised_sum = 0.0
        physics_sum = 0.0
        identity_sum = 0.0
        elimination_sum = 0.0
        sample_count = 0
        epoch_started = time.time()
        for batch_index, (u_batch, v_batch, target_norm) in enumerate(
            train_loader
        ):
            if args.max_batches and batch_index >= args.max_batches:
                break
            u_batch = u_batch.to(device, non_blocking=True)
            v_batch = v_batch.to(device, non_blocking=True)
            target_norm = target_norm.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if soft_feature_gradient_scale < 1.0:
                captured_head_input = {}

                def capture_head_input(module, inputs):
                    del module
                    captured_head_input["value"] = inputs[0]

                hook = model.fc_out.register_forward_pre_hook(
                    capture_head_input
                )
                predictions_norm = model(u_batch)
                hook.remove()
            else:
                predictions_norm = model(u_batch)
            supervised_loss = (
                (predictions_norm - target_norm).square()
                * supervised_weights[None]
            ).mean()
            if (
                lambda_physics > 0.0
                or lambda_identity > 0.0
                or lambda_elimination > 0.0
            ):
                predictions_real = (
                    train_set.scaler.inverse_transform_tensor(
                        predictions_norm
                    )
                )
            if lambda_physics > 0.0:
                physical_loss = physics_loss_fn(
                    u_batch,
                    v_batch,
                    predictions_real,
                )
            else:
                physical_loss = predictions_norm.new_zeros(())
            if lambda_identity > 0.0 or lambda_elimination > 0.0:
                soft_predictions_real = predictions_real
                if soft_feature_gradient_scale < 1.0:
                    # The analytic constraints train only the existing
                    # parameter head when scale=0, or send a controlled
                    # fraction of their gradient into the shared features.
                    head_input = captured_head_input["value"]
                    soft_head_input = (
                        head_input.detach()
                        + soft_feature_gradient_scale
                        * (head_input - head_input.detach())
                    )
                    soft_predictions_norm = torch.sigmoid(
                        model.fc_out(soft_head_input)
                    )
                    soft_predictions_real = (
                        train_set.scaler.inverse_transform_tensor(
                            soft_predictions_norm
                        )
                    )
                identity_loss, elimination_loss = (
                    soft_constraint_loss_fn(
                        u_batch,
                        soft_predictions_real,
                    )
                )
            else:
                identity_loss = predictions_norm.new_zeros(())
                elimination_loss = predictions_norm.new_zeros(())
            total_loss = (
                supervised_loss
                + lambda_physics * physical_loss
                + lambda_identity * identity_loss
                + lambda_elimination * elimination_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, "
                    f"batch {batch_index + 1}."
                )
            total_loss.backward()
            optimizer.step()

            count = len(u_batch)
            sample_count += count
            supervised_sum += float(supervised_loss.detach()) * count
            physics_sum += float(physical_loss.detach()) * count
            identity_sum += float(identity_loss.detach()) * count
            elimination_sum += (
                float(elimination_loss.detach()) * count
            )

        epoch_record = {
            "epoch": epoch + 1,
            "learning_rate": lr,
            "lambda_physics": lambda_physics,
            "lambda_identity": lambda_identity,
            "lambda_elimination": lambda_elimination,
            "train_supervised_mse": supervised_sum / sample_count,
            "train_physics_loss": physics_sum / sample_count,
            "train_identity_loss": identity_sum / sample_count,
            "train_elimination_loss": elimination_sum / sample_count,
            "seconds": time.time() - epoch_started,
        }
        if validation_loader is not None:
            validation_metrics = evaluate(
                model,
                validation_loader,
                device,
                train_set.parameter_names,
                train_set.scaler,
            )
            epoch_record["validation_normalized_rmse"] = (
                validation_metrics["normalized_rmse"]
            )
            epoch_record["validation_normalized_mae"] = (
                validation_metrics["normalized_mae"]
            )
            if (
                validation_metrics["normalized_rmse"]
                < best_validation_rmse
            ):
                best_validation_rmse = validation_metrics[
                    "normalized_rmse"
                ]
                best_epoch = epoch + 1
                best_state = copy.deepcopy(
                    {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    }
                )
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            train_set.parameter_names,
            train_set.scaler,
        )
    else:
        best_epoch = args.epochs
        validation_metrics = None
    metrics = evaluate(
        model,
        eval_loader,
        device,
        train_set.parameter_names,
        train_set.scaler,
    )
    result = {
        "run_name": run_name,
        "pde": args.pde,
        "loss_type": args.loss_type,
        "seed": args.seed,
        "model_arch": args.model_arch,
        "model": {
            "model_arch": args.model_arch,
            "nk": args.nk,
            "kernel_size": args.kernel_size,
            "hidden_features": args.hidden_features,
            "output_dim": 3,
            "input_field": "u",
        },
        "training": {
            "train_data": str(args.train_data),
            "validation_data": (
                str(args.train_data)
                if validation_set is not None
                else None
            ),
            "eval_data": str(args.eval_data),
            "train_samples": len(train_set),
            "validation_samples": (
                len(validation_set)
                if validation_set is not None
                else 0
            ),
            "eval_samples": len(eval_set),
            "validation_count": args.validation_count,
            "validation_seed": (
                args.validation_seed
                if args.validation_count
                else None
            ),
            "validation_split_file": (
                str(split_path) if split_path is not None else None
            ),
            "selection_metric": (
                "validation_normalized_rmse"
                if validation_set is not None
                else "final_epoch"
            ),
            "best_epoch": best_epoch,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "lambda_physics": (
                args.lambda_physics
                if args.loss_type == "physical"
                or args.loss_type == "soft_physical"
                else 0.0
            ),
            "physics_start_epoch": args.physics_start_epoch,
            "physics_masked": not args.no_physics_mask,
            "lambda_identity": (
                args.lambda_identity
                if args.loss_type == "soft_physical"
                else 0.0
            ),
            "lambda_elimination": (
                args.lambda_elimination
                if args.loss_type == "soft_physical"
                else 0.0
            ),
            "soft_constraint_start_epoch": (
                args.soft_constraint_start_epoch
                if args.loss_type == "soft_physical"
                else None
            ),
            "soft_constraint_ramp_epochs": (
                args.soft_constraint_ramp_epochs
                if args.loss_type == "soft_physical"
                else None
            ),
            "hard_projection_at_inference": False,
            "soft_constraints_update": (
                "parameter_head_only"
                if soft_feature_gradient_scale == 0.0
                else (
                    "entire_cnn"
                    if soft_feature_gradient_scale == 1.0
                    else "scaled_shared_features"
                )
            ),
            "soft_feature_gradient_scale": (
                soft_feature_gradient_scale
            ),
            "supervised_weights": supervised_weights.tolist(),
            "duration_seconds": time.time() - started_at,
        },
        "validation": validation_metrics,
        "parameter_names": list(train_set.parameter_names),
        "scaler": {
            "data_min": train_set.scaler.min_val.tolist(),
            "data_max": train_set.scaler.max_val.tolist(),
        },
        "evaluation": metrics,
        "checkpoint": str(checkpoint_path),
    }
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "model_config": result["model"],
            "training_config": result["training"],
            "parameter_names": result["parameter_names"],
            "scaler": train_set.scaler.state_dict(),
            "validation": validation_metrics,
            "evaluation": metrics,
        },
        checkpoint_path,
    )
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a controlled pure-vs-physical PDE experiment."
        )
    )
    parser.add_argument(
        "--pde",
        choices=("brusselator", "schnakenberg", "lengyel_epstein"),
        default="brusselator",
    )
    parser.add_argument(
        "--loss-type",
        choices=("pure", "physical", "soft_physical"),
        required=True,
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-arch",
        choices=MODEL_ARCHES,
        default="cnn1",
    )
    parser.add_argument("--nk", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--hidden-features", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--lambda-physics", type=float, default=1e-2)
    parser.add_argument(
        "--supervised-weights",
        default="1,1,1",
        help=(
            "Comma-separated per-parameter supervised MSE weights; "
            "the values are normalized to mean one."
        ),
    )
    parser.add_argument("--physics-start-epoch", type=int, default=20)
    parser.add_argument("--lambda-identity", type=float, default=0.0)
    parser.add_argument("--lambda-elimination", type=float, default=0.0)
    parser.add_argument(
        "--soft-constraint-start-epoch",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--soft-constraint-ramp-epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--soft-head-only",
        action="store_true",
        help=(
            "Backpropagate analytic soft constraints only through the "
            "existing final parameter head. Supervised and original PDE "
            "losses still update the complete CNN."
        ),
    )
    parser.add_argument(
        "--soft-feature-gradient-scale",
        type=float,
        default=1.0,
        help=(
            "Fraction of analytic soft-constraint gradient sent into "
            "shared CNN features (0=head only, 1=full CNN). This does "
            "not change inference."
        ),
    )
    parser.add_argument(
        "--no-physics-mask",
        action="store_true",
        help="Evaluate the physical residual over every pixel.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-count",
        type=int,
        default=4000,
        help=(
            "Reserve this many samples from --train-data for validation "
            "checkpoint selection; --eval-data remains the final test set."
        ),
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=42,
        help="Fixed seed used only for the train/validation split.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit batches per epoch for a smoke test; 0 uses all batches.",
    )
    args = parser.parse_args()
    try:
        args.supervised_weights = tuple(
            float(value)
            for value in args.supervised_weights.split(",")
        )
    except ValueError:
        parser.error("--supervised-weights must contain numbers.")
    if (
        len(args.supervised_weights) != 3
        or any(value <= 0 for value in args.supervised_weights)
    ):
        parser.error(
            "--supervised-weights must contain three positive values."
        )
    defaults = {
        "brusselator": {
            "train_data": Path(
                "brusselator_data/brusselator_train_data.npz"
            ),
            "eval_data": Path(
                "brusselator_data/brusselator_eval_data.npz"
            ),
            "output_dir": Path(
                "standardized_split_experiments/brusselator"
            ),
        },
        "schnakenberg": {
            "train_data": Path(
                "schnakenberg_data/schnakenberg_train_data.npz"
            ),
            "eval_data": Path(
                "schnakenberg_data/schnakenberg_eval_data.npz"
            ),
            "output_dir": Path(
                "standardized_split_experiments/schnakenberg"
            ),
        },
        "lengyel_epstein": {
            "train_data": Path(
                "lengyel_epstein_data/lengyel_epstein_train_data.npz"
            ),
            "eval_data": Path(
                "lengyel_epstein_data/lengyel_epstein_eval_data.npz"
            ),
            "output_dir": Path(
                "standardized_split_experiments/lengyel_epstein"
            ),
        },
    }[args.pde]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive.")
    if args.validation_count < 0:
        parser.error("--validation-count must be non-negative.")
    if args.lambda_physics < 0:
        parser.error("--lambda-physics must be non-negative.")
    if args.lambda_identity < 0 or args.lambda_elimination < 0:
        parser.error(
            "--lambda-identity and --lambda-elimination "
            "must be non-negative."
        )
    if args.soft_head_only and args.loss_type != "soft_physical":
        parser.error(
            "--soft-head-only is only valid with --loss-type soft_physical."
        )
    if not 0.0 <= args.soft_feature_gradient_scale <= 1.0:
        parser.error(
            "--soft-feature-gradient-scale must be between 0 and 1."
        )
    if args.soft_head_only and args.soft_feature_gradient_scale != 1.0:
        parser.error(
            "Use either --soft-head-only or "
            "--soft-feature-gradient-scale, not both."
        )
    if args.physics_start_epoch < 0:
        parser.error("--physics-start-epoch must be non-negative.")
    if (
        args.soft_constraint_start_epoch < 0
        or args.soft_constraint_ramp_epochs <= 0
    ):
        parser.error(
            "Soft constraint start must be non-negative and "
            "ramp epochs must be positive."
        )
    for path in (args.train_data, args.eval_data):
        if not path.is_file():
            parser.error(f"Dataset does not exist: {path}")
    return args


if __name__ == "__main__":
    train(parse_args())
