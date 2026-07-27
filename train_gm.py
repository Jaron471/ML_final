#!/usr/bin/env python3
"""Controlled Pure/Physical/Soft-Physical Gierer--Meinhardt training."""

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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from model.dataset import TuringDataset
from model.loss import PhysicsLoss
from model.model import MODEL_ARCHES, build_model
from model.soft_constraints import GMSoftConstraintLoss


PARAMETER_NAMES = ("a", "b", "c", "delta")


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


def ramp_weight(
    active: bool,
    epoch: int,
    maximum: float,
    start_epoch: int,
    ramp_epochs: int,
) -> float:
    if not active or maximum <= 0.0 or epoch < start_epoch:
        return 0.0
    progress = (epoch - start_epoch + 1) / max(1, ramp_epochs)
    return maximum * min(1.0, progress)


def calculate_metrics(
    targets_norm: np.ndarray,
    predictions_norm: np.ndarray,
    scaler,
) -> dict:
    targets_real = scaler.inverse_transform_numpy(targets_norm)
    predictions_real = scaler.inverse_transform_numpy(predictions_norm)
    error_norm = predictions_norm - targets_norm
    error_real = predictions_real - targets_real
    rmse_norm = np.sqrt(np.mean(np.square(error_norm), axis=0))
    mae_norm = np.mean(np.abs(error_norm), axis=0)
    rmse_real = np.sqrt(np.mean(np.square(error_real), axis=0))
    mae_real = np.mean(np.abs(error_real), axis=0)
    ss_res = np.sum(np.square(error_real), axis=0)
    ss_tot = np.sum(
        (
            targets_real
            - np.mean(targets_real, axis=0)
        ) ** 2,
        axis=0,
    )
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    per_parameter = {}
    for index, name in enumerate(PARAMETER_NAMES):
        per_parameter[name] = {
            "rmse_real": float(rmse_real[index]),
            "mae_real": float(mae_real[index]),
            "rmse_normalized": float(rmse_norm[index]),
            "mae_normalized": float(mae_norm[index]),
            "r2": float(r2[index]),
        }
    normalized_rmse = float(
        np.sqrt(np.mean(np.square(error_norm)))
    )
    normalized_mean_norm = float(
        np.linalg.norm(np.mean(targets_norm, axis=0))
    )
    return {
        "normalized_rmse": normalized_rmse,
        "normalized_mae": float(np.mean(np.abs(error_norm))),
        "paper_joint_nrmse": (
            normalized_rmse / normalized_mean_norm
            if normalized_mean_norm > 0.0
            else 0.0
        ),
        "mean_r2": float(np.mean(r2)),
        "per_parameter": per_parameter,
    }


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets = []
    predictions = []
    for u_batch, _, target_batch in loader:
        prediction = model(
            u_batch.to(device, non_blocking=True)
        )
        targets.append(target_batch.numpy())
        predictions.append(prediction.cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_set = TuringDataset(
        args.train_data,
        parameter_names=PARAMETER_NAMES,
    )
    validation_set = TuringDataset(
        args.validation_data,
        scaler=train_set.scaler,
        parameter_names=PARAMETER_NAMES,
    )
    evaluation_set = TuringDataset(
        args.eval_data,
        scaler=train_set.scaler,
        parameter_names=PARAMETER_NAMES,
    )
    train_set.scaler.to_device(device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    evaluation_loader = DataLoader(
        evaluation_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model_arch,
        nk=args.nk,
        np_size=args.kernel_size,
        nf=args.hidden_features,
        input_size=128,
        output_dim=4,
    ).to(device)
    if (
        args.soft_feature_gradient_scale < 1.0
        and not hasattr(model, "fc_out")
    ):
        raise ValueError(
            "Partial soft gradients require a CNN with an fc_out head."
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )
    physics_loss_fn = PhysicsLoss(
        dx=1.0,
        s_diffusion=0.4,
    ).to(device)
    soft_loss_fn = GMSoftConstraintLoss(
        grid_size=128,
        dx=1.0,
        s_diffusion=0.4,
    ).to(device)
    supervised_weights = torch.tensor(
        args.supervised_weights,
        dtype=torch.float32,
        device=device,
    )
    supervised_weights = (
        supervised_weights / supervised_weights.mean()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"gm_{args.model_arch}_{args.loss_type}_seed{args.seed}"
    )
    checkpoint_path = args.output_dir / f"{run_name}.pt"
    history_path = args.output_dir / f"{run_name}_history.json"
    metrics_path = args.output_dir / f"{run_name}_metrics.json"

    best_epoch = 0
    best_validation_rmse = float("inf")
    best_state = None
    history = []
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
                "validation_samples": len(validation_set),
                "eval_samples": len(evaluation_set),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "supervised_weights": supervised_weights.tolist(),
            }
        ),
        flush=True,
    )

    uses_physics = args.loss_type in (
        "physical",
        "soft_physical",
    )
    uses_soft = args.loss_type == "soft_physical"
    for epoch in range(args.epochs):
        model.train()
        learning_rate = learning_rate_at_epoch(
            epoch,
            args.epochs,
            args.learning_rate,
            args.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        lambda_physics = (
            args.lambda_physics
            if uses_physics and epoch >= args.physics_start_epoch
            else 0.0
        )
        lambda_integrated = ramp_weight(
            uses_soft,
            epoch,
            args.lambda_integrated,
            args.soft_start_epoch,
            args.soft_ramp_epochs,
        )
        lambda_delta = ramp_weight(
            uses_soft,
            epoch,
            args.lambda_delta,
            args.soft_start_epoch,
            args.soft_ramp_epochs,
        )

        sums = {
            "supervised": 0.0,
            "physics": 0.0,
            "integrated": 0.0,
            "delta": 0.0,
        }
        sample_count = 0
        epoch_started = time.time()
        for batch_index, (
            u_batch,
            v_batch,
            target_norm,
        ) in enumerate(train_loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            u_batch = u_batch.to(device, non_blocking=True)
            v_batch = v_batch.to(device, non_blocking=True)
            target_norm = target_norm.to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)

            if (
                uses_soft
                and args.soft_feature_gradient_scale < 1.0
            ):
                captured = {}

                def capture_head_input(module, inputs):
                    del module
                    captured["value"] = inputs[0]

                hook = model.fc_out.register_forward_pre_hook(
                    capture_head_input
                )
                predictions_norm = model(u_batch)
                hook.remove()
            else:
                predictions_norm = model(u_batch)

            supervised_loss = (
                (
                    predictions_norm - target_norm
                ).square()
                * supervised_weights[None]
            ).mean()
            if (
                lambda_physics > 0.0
                or lambda_integrated > 0.0
                or lambda_delta > 0.0
            ):
                predictions_real = (
                    train_set.scaler.inverse_transform_tensor(
                        predictions_norm
                    )
                )
            if lambda_physics > 0.0:
                physics_loss = physics_loss_fn(
                    u_batch,
                    v_batch,
                    predictions_real,
                )
            else:
                physics_loss = predictions_norm.new_zeros(())

            if lambda_integrated > 0.0 or lambda_delta > 0.0:
                soft_predictions_real = predictions_real
                if args.soft_feature_gradient_scale < 1.0:
                    head_input = captured["value"]
                    scale = args.soft_feature_gradient_scale
                    routed_head_input = (
                        head_input.detach()
                        + scale
                        * (head_input - head_input.detach())
                    )
                    soft_predictions_norm = torch.sigmoid(
                        model.fc_out(routed_head_input)
                    )
                    soft_predictions_real = (
                        train_set.scaler.inverse_transform_tensor(
                            soft_predictions_norm
                        )
                    )
                integrated_loss, delta_loss = soft_loss_fn(
                    u_batch,
                    v_batch,
                    soft_predictions_real,
                )
            else:
                integrated_loss = predictions_norm.new_zeros(())
                delta_loss = predictions_norm.new_zeros(())

            total_loss = (
                supervised_loss
                + lambda_physics * physics_loss
                + lambda_integrated * integrated_loss
                + lambda_delta * delta_loss
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
            sums["supervised"] += (
                float(supervised_loss.detach()) * count
            )
            sums["physics"] += (
                float(physics_loss.detach()) * count
            )
            sums["integrated"] += (
                float(integrated_loss.detach()) * count
            )
            sums["delta"] += float(delta_loss.detach()) * count

        validation_targets, validation_predictions = predict(
            model,
            validation_loader,
            device,
        )
        validation_metrics = calculate_metrics(
            validation_targets,
            validation_predictions,
            train_set.scaler,
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

        record = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "lambda_physics": lambda_physics,
            "lambda_integrated": lambda_integrated,
            "lambda_delta": lambda_delta,
            "train_supervised_mse": (
                sums["supervised"] / sample_count
            ),
            "train_physics_loss": (
                sums["physics"] / sample_count
            ),
            "train_integrated_loss": (
                sums["integrated"] / sample_count
            ),
            "train_delta_loss": sums["delta"] / sample_count,
            "validation_normalized_rmse": (
                validation_metrics["normalized_rmse"]
            ),
            "seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    model.load_state_dict(best_state)
    validation_targets, validation_predictions = predict(
        model,
        validation_loader,
        device,
    )
    evaluation_targets, evaluation_predictions = predict(
        model,
        evaluation_loader,
        device,
    )
    validation_metrics = calculate_metrics(
        validation_targets,
        validation_predictions,
        train_set.scaler,
    )
    evaluation_metrics = calculate_metrics(
        evaluation_targets,
        evaluation_predictions,
        train_set.scaler,
    )
    training_config = {
        "loss_type": args.loss_type,
        "train_data": str(args.train_data),
        "validation_data": str(args.validation_data),
        "eval_data": str(args.eval_data),
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "eval_samples": len(evaluation_set),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "lambda_physics": (
            args.lambda_physics if uses_physics else 0.0
        ),
        "physics_start_epoch": args.physics_start_epoch,
        "lambda_integrated": (
            args.lambda_integrated if uses_soft else 0.0
        ),
        "lambda_delta": (
            args.lambda_delta if uses_soft else 0.0
        ),
        "soft_start_epoch": (
            args.soft_start_epoch if uses_soft else None
        ),
        "soft_ramp_epochs": (
            args.soft_ramp_epochs if uses_soft else None
        ),
        "soft_feature_gradient_scale": (
            args.soft_feature_gradient_scale
            if uses_soft
            else None
        ),
        "supervised_weights": supervised_weights.tolist(),
        "hard_projection_at_inference": False,
        "duration_seconds": time.time() - started_at,
    }
    model_config = {
        "model_arch": args.model_arch,
        "nk": args.nk,
        "kernel_size": args.kernel_size,
        "hidden_features": args.hidden_features,
        "input_size": 128,
        "output_dim": 4,
        "input_field": "u",
    }
    result = {
        "run_name": run_name,
        "pde": "gierer_meinhardt",
        "seed": args.seed,
        "model": model_config,
        "training": training_config,
        "parameter_names": list(PARAMETER_NAMES),
        "scaler": {
            "data_min": train_set.scaler.min_val.tolist(),
            "data_max": train_set.scaler.max_val.tolist(),
        },
        "validation": validation_metrics,
        "evaluation": evaluation_metrics,
        "checkpoint": str(checkpoint_path),
    }
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": best_state,
            "model_config": model_config,
            "training_config": training_config,
            "parameter_names": list(PARAMETER_NAMES),
            "scaler": train_set.scaler.state_dict(),
            "validation": validation_metrics,
            "evaluation": evaluation_metrics,
        },
        checkpoint_path,
    )
    history_path.write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loss-type",
        choices=("pure", "physical", "soft_physical"),
        required=True,
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("gm_data/gm_train_data.npz"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("gm_data/gm_validation_data.npz"),
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
    parser.add_argument("--lambda-physics", type=float, default=0.01)
    parser.add_argument("--physics-start-epoch", type=int, default=20)
    parser.add_argument("--lambda-integrated", type=float, default=0.1)
    parser.add_argument("--lambda-delta", type=float, default=0.1)
    parser.add_argument("--soft-start-epoch", type=int, default=20)
    parser.add_argument("--soft-ramp-epochs", type=int, default=20)
    parser.add_argument(
        "--soft-feature-gradient-scale",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--supervised-weights",
        default="1,1,1,1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    try:
        args.supervised_weights = tuple(
            float(value)
            for value in args.supervised_weights.split(",")
        )
    except ValueError:
        parser.error("--supervised-weights must contain numbers.")
    if (
        len(args.supervised_weights) != 4
        or any(value <= 0 for value in args.supervised_weights)
    ):
        parser.error(
            "--supervised-weights must contain four positive values."
        )
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive.")
    for name in (
        "lambda_physics",
        "lambda_integrated",
        "lambda_delta",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")
    if (
        args.physics_start_epoch < 0
        or args.soft_start_epoch < 0
        or args.soft_ramp_epochs <= 0
    ):
        parser.error("Loss schedule arguments are invalid.")
    if not 0.0 <= args.soft_feature_gradient_scale <= 1.0:
        parser.error(
            "--soft-feature-gradient-scale must be between 0 and 1."
        )
    for path in (
        args.train_data,
        args.validation_data,
        args.eval_data,
    ):
        if not path.is_file():
            parser.error(f"Dataset does not exist: {path}")
    return args


if __name__ == "__main__":
    train(parse_args())
