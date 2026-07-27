#!/usr/bin/env python3
"""Generate a converged 2-D three-parameter Brusselator dataset.

The nondimensional system is

    du/dt = Laplacian(u) + gamma * (A - (B + 1)u + u^2 v)
    dv/dt = d * Laplacian(v) + gamma * (B u - u^2 v)

gamma and D_u are fixed, so the inferred parameters are [A, B, d].
Only samples that pass a finite-grid Turing test and an a-posteriori
spectral steady-PDE-residual test are retained.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenerationConfig:
    total_samples: int = 20_000
    train_samples: int = 16_000
    eval_samples: int = 4_000
    grid_size: int = 128
    dx: float = 1.0
    dt: float = 0.1
    gamma: float = 1.0
    max_steps: int = 50_000
    min_steps: int = 2_000
    check_interval: int = 400
    consecutive_checks: int = 3
    noise_std: float = 0.03
    convergence_rms: float = 1e-4
    min_pattern_std: float = 1e-3
    batch_size: int = 128
    parameter_seed: int = 20_260_724
    split_seed: int = 42
    noise_seed: int = 43
    A_min: float = 0.5
    A_max: float = 1.5
    B_fraction_min: float = 0.1
    B_fraction_max: float = 0.9
    B_min: float = 1.25
    d_min: float = 8.0
    d_max: float = 60.0
    growth_min: float = 0.02
    growth_max: float = 0.2
    wavelength_min: float = 8.0
    wavelength_max: float = 64.0


SAMPLE_KEYS = (
    "u",
    "v",
    "A",
    "B",
    "d",
    "u_star",
    "v_star",
    "final_step",
    "residual_rms",
    "pattern_std",
    "linear_growth_rate",
    "dominant_wavelength",
    "noise_seed",
    "converged",
)


def sample_turing_parameters(
    rng: np.random.Generator, count: int, cfg: GenerationConfig
) -> np.ndarray:
    """Sample [A,B,d] in the homogeneous-stable Turing region."""
    accepted: list[np.ndarray] = []
    accepted_count = 0
    q = 2.0 * np.pi * np.arange(1, cfg.grid_size // 2 + 1) / (
        cfg.grid_size * cfg.dx
    )
    q2 = q * q

    while accepted_count < count:
        draw_count = max(8 * (count - accepted_count), 512)
        A = rng.uniform(cfg.A_min, cfg.A_max, draw_count)
        # B = 1 + r A^2 guarantees B < 1 + A^2 for r < 1, so
        # the spatially homogeneous Brusselator steady state is stable.
        fraction = rng.uniform(
            cfg.B_fraction_min, cfg.B_fraction_max, draw_count
        )
        B = 1.0 + fraction * A**2
        d = np.exp(
            rng.uniform(np.log(cfg.d_min), np.log(cfg.d_max), draw_count)
        )

        f_u = B - 1.0
        f_v = A**2
        g_u = -B
        g_v = -(A**2)
        trace_j = cfg.gamma * (f_u + g_v)
        det_j = (cfg.gamma**2) * (f_u * g_v - f_v * g_u)
        diffusion_trace = cfg.gamma * (d * f_u + g_v)
        turing_margin = diffusion_trace**2 - 4.0 * d * det_j
        analytic_ok = (
            (trace_j < 0.0)
            & (det_j > 0.0)
            & (diffusion_trace > 0.0)
            & (turing_margin > 0.0)
        )

        trace_q = trace_j[:, None] - (1.0 + d[:, None]) * q2
        det_q = (
            det_j[:, None]
            - diffusion_trace[:, None] * q2
            + d[:, None] * q2**2
        )
        discriminant = trace_q**2 - 4.0 * det_q
        largest_real_eigenvalue = np.where(
            discriminant >= 0.0,
            0.5 * (trace_q + np.sqrt(np.maximum(discriminant, 0.0))),
            0.5 * trace_q,
        )
        mode_index = np.argmax(largest_real_eigenvalue, axis=1)
        growth_rate = largest_real_eigenvalue[
            np.arange(draw_count), mode_index
        ]
        dominant_wavelength = 2.0 * np.pi / np.sqrt(q2[mode_index])
        keep = (
            analytic_ok
            & (B >= cfg.B_min)
            & (growth_rate >= cfg.growth_min)
            & (growth_rate <= cfg.growth_max)
            & (dominant_wavelength >= cfg.wavelength_min)
            & (dominant_wavelength <= cfg.wavelength_max)
        )
        batch = np.stack(
            (
                A[keep],
                B[keep],
                d[keep],
                growth_rate[keep],
                dominant_wavelength[keep],
            ),
            axis=1,
        )
        if len(batch):
            accepted.append(batch)
            accepted_count += len(batch)

    return np.concatenate(accepted, axis=0)[:count].astype(np.float32)


def make_initial_noise(
    seeds: np.ndarray, grid_size: int, noise_std: float
) -> tuple[np.ndarray, np.ndarray]:
    u_noise = np.empty((len(seeds), grid_size, grid_size), dtype=np.float32)
    v_noise = np.empty_like(u_noise)
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        u_noise[index] = rng.normal(0.0, noise_std, u_noise.shape[1:])
        v_noise[index] = rng.normal(0.0, noise_std, v_noise.shape[1:])
    return u_noise, v_noise


def simulate_batch(
    params: np.ndarray,
    noise_seeds: np.ndarray,
    gpu_id: int,
    cfg: GenerationConfig,
) -> dict[str, np.ndarray]:
    batch_size = len(params)
    u_noise, v_noise = make_initial_noise(
        noise_seeds, cfg.grid_size, cfg.noise_std
    )

    with cp.cuda.Device(gpu_id):
        A = cp.asarray(params[:, 0, None, None], dtype=cp.float32)
        B = cp.asarray(params[:, 1, None, None], dtype=cp.float32)
        d = cp.asarray(params[:, 2, None, None], dtype=cp.float32)
        u_star = A
        v_star = B / A
        u = u_star + cp.asarray(u_noise)
        v = v_star + cp.asarray(v_noise)

        k = 2.0 * cp.pi * cp.fft.fftfreq(cfg.grid_size, d=cfg.dx)
        laplace_eigenvalue = -(
            k[:, None] ** 2 + k[None, :] ** 2
        ).astype(cp.float32)
        denominator_u = 1.0 - cfg.dt * laplace_eigenvalue
        denominator_v = 1.0 - cfg.dt * d * laplace_eigenvalue
        u_hat = cp.fft.fft2(u)
        v_hat = cp.fft.fft2(v)
        converged = cp.zeros(batch_size, dtype=cp.bool_)
        failed = cp.zeros(batch_size, dtype=cp.bool_)
        streak = cp.zeros(batch_size, dtype=cp.int32)
        final_step = cp.zeros(batch_size, dtype=cp.int32)

        for step in range(1, cfg.max_steps + 1):
            active = ~(converged | failed)
            if not bool(cp.any(active)):
                break

            uv2 = u * u * v
            reaction_u = cfg.gamma * (A - (B + 1.0) * u + uv2)
            reaction_v = cfg.gamma * (B * u - uv2)
            next_u_hat = (
                u_hat + cfg.dt * cp.fft.fft2(reaction_u)
            ) / denominator_u
            next_v_hat = (
                v_hat + cfg.dt * cp.fft.fft2(reaction_v)
            ) / denominator_v
            next_u = cp.fft.ifft2(next_u_hat).real.astype(cp.float32)
            next_v = cp.fft.ifft2(next_v_hat).real.astype(cp.float32)

            active_3d = active[:, None, None]
            next_u = cp.where(active_3d, next_u, u)
            next_v = cp.where(active_3d, next_v, v)
            next_u_hat = cp.where(active_3d, next_u_hat, u_hat)
            next_v_hat = cp.where(active_3d, next_v_hat, v_hat)

            if step % cfg.check_interval == 0:
                update_rms = cp.sqrt(
                    cp.mean(
                        (next_u - u) ** 2 + (next_v - v) ** 2,
                        axis=(1, 2),
                    )
                    / 2.0
                ) / cfg.dt
                pattern_std = cp.std(next_u, axis=(1, 2))
                finite = cp.all(cp.isfinite(next_u), axis=(1, 2)) & cp.all(
                    cp.isfinite(next_v), axis=(1, 2)
                )
                bounded = (
                    (cp.max(cp.abs(next_u), axis=(1, 2)) < 100.0)
                    & (cp.max(cp.abs(next_v), axis=(1, 2)) < 100.0)
                )
                failed |= active & ~(finite & bounded)
                small_residual = (
                    (step >= cfg.min_steps)
                    & (update_rms <= cfg.convergence_rms)
                    & (pattern_std >= cfg.min_pattern_std)
                    & finite
                    & bounded
                )
                streak = cp.where(active & small_residual, streak + 1, 0)
                newly_converged = active & (
                    streak >= cfg.consecutive_checks
                )
                final_step = cp.where(
                    newly_converged, step, final_step
                )
                converged |= newly_converged

            u, v = next_u, next_v
            u_hat, v_hat = next_u_hat, next_v_hat

        uv2 = u * u * v
        lap_u = cp.fft.ifft2(laplace_eigenvalue * u_hat).real
        lap_v = cp.fft.ifft2(laplace_eigenvalue * v_hat).real
        residual_u = lap_u + cfg.gamma * (
            A - (B + 1.0) * u + uv2
        )
        residual_v = d * lap_v + cfg.gamma * (B * u - uv2)
        residual_rms = cp.sqrt(
            cp.mean(residual_u**2 + residual_v**2, axis=(1, 2)) / 2.0
        )
        pattern_std = cp.std(u, axis=(1, 2))
        finite = cp.all(cp.isfinite(u), axis=(1, 2)) & cp.all(
            cp.isfinite(v), axis=(1, 2)
        )
        exact_converged = (
            finite
            & ~failed
            & (residual_rms <= cfg.convergence_rms)
            & (pattern_std >= cfg.min_pattern_std)
            & (cp.min(u, axis=(1, 2)) >= 0.0)
            & (cp.min(v, axis=(1, 2)) >= 0.0)
        )
        final_step = cp.where(
            exact_converged & (final_step == 0),
            cfg.max_steps,
            final_step,
        )

        result = {
            "u": cp.asnumpy(u).astype(np.float32),
            "v": cp.asnumpy(v).astype(np.float32),
            "A": params[:, 0].astype(np.float32),
            "B": params[:, 1].astype(np.float32),
            "d": params[:, 2].astype(np.float32),
            "u_star": cp.asnumpy(u_star[:, 0, 0]).astype(np.float32),
            "v_star": cp.asnumpy(v_star[:, 0, 0]).astype(np.float32),
            "final_step": cp.asnumpy(final_step).astype(np.int32),
            "residual_rms": cp.asnumpy(residual_rms).astype(np.float32),
            "pattern_std": cp.asnumpy(pattern_std).astype(np.float32),
            "linear_growth_rate": params[:, 3].astype(np.float32),
            "dominant_wavelength": params[:, 4].astype(np.float32),
            "noise_seed": noise_seeds.astype(np.uint64),
            "converged": cp.asnumpy(exact_converged),
        }
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return result


def worker_generate(
    gpu_id: int,
    target_samples: int,
    output_path: str,
    cfg: GenerationConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.parameter_seed + 100_003 * gpu_id)
    accepted_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in SAMPLE_KEYS
    }
    accepted_count = 0
    attempted_count = 0
    candidate_index = 0

    while accepted_count < target_samples:
        current_batch_size = min(
            cfg.batch_size, target_samples - accepted_count + 32
        )
        params = sample_turing_parameters(rng, current_batch_size, cfg)
        local_indices = np.arange(
            candidate_index,
            candidate_index + current_batch_size,
            dtype=np.uint64,
        )
        noise_seeds = (
            np.uint64(cfg.noise_seed)
            + np.uint64(gpu_id) * np.uint64(1_000_000_000)
            + local_indices
        )
        candidate_index += current_batch_size
        attempted_count += current_batch_size
        result = simulate_batch(params, noise_seeds, gpu_id, cfg)
        valid_indices = np.flatnonzero(result["converged"])
        needed = target_samples - accepted_count
        valid_indices = valid_indices[:needed]
        for key in SAMPLE_KEYS:
            accepted_parts[key].append(result[key][valid_indices])
        accepted_count += len(valid_indices)
        print(
            f"[GPU {gpu_id}] accepted={accepted_count}/{target_samples}, "
            f"attempted={attempted_count}, batch_pass={len(valid_indices)}",
            flush=True,
        )

    shard = {
        key: np.concatenate(parts, axis=0)
        for key, parts in accepted_parts.items()
    }
    np.savez(output_path, **shard)
    return {
        "gpu_id": gpu_id,
        "path": output_path,
        "accepted": accepted_count,
        "attempted": attempted_count,
    }


def merge_split_validate(
    worker_summaries: list[dict[str, Any]],
    cfg: GenerationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    loaded = [
        np.load(Path(summary["path"])) for summary in worker_summaries
    ]
    merged = {
        key: np.concatenate([data[key] for data in loaded], axis=0)[
            : cfg.total_samples
        ]
        for key in SAMPLE_KEYS
    }
    for data in loaded:
        data.close()
    merged["ids"] = np.arange(
        1, cfg.total_samples + 1, dtype=np.int32
    )

    prefix = f"brusselator_{cfg.total_samples}"
    params_path = output_dir / f"{prefix}_params.csv"
    pd.DataFrame(
        {
            "id": merged["ids"],
            "A": merged["A"],
            "B": merged["B"],
            "d": merged["d"],
        }
    ).to_csv(params_path, index=False)
    report_path = output_dir / "brusselator_convergence_report.csv"
    pd.DataFrame(
        {
            "id": merged["ids"],
            "final_step": merged["final_step"],
            "residual_rms": merged["residual_rms"],
            "pattern_std": merged["pattern_std"],
            "linear_growth_rate": merged["linear_growth_rate"],
            "dominant_wavelength": merged["dominant_wavelength"],
            "noise_seed": merged["noise_seed"],
            "converged": merged["converged"],
        }
    ).to_csv(report_path, index=False)

    metadata = {
        "pde_name": np.array("brusselator"),
        "parameter_names": np.array(["A", "B", "d"]),
        "gamma": np.float32(cfg.gamma),
        "grid_size": np.int32(cfg.grid_size),
        "dx": np.float32(cfg.dx),
        "dt": np.float32(cfg.dt),
        "parameter_seed": np.int64(cfg.parameter_seed),
        "split_seed": np.int64(cfg.split_seed),
    }
    full_path = output_dir / f"{prefix}_dataset.npz"
    np.savez(full_path, **merged, **metadata)

    permutation = np.random.default_rng(cfg.split_seed).permutation(
        cfg.total_samples
    )
    train_indices = permutation[: cfg.train_samples]
    eval_indices = permutation[cfg.train_samples :]
    train_path = output_dir / "brusselator_train_data.npz"
    eval_path = output_dir / "brusselator_eval_data.npz"
    np.savez(
        train_path,
        **{key: value[train_indices] for key, value in merged.items()},
        **metadata,
    )
    np.savez(
        eval_path,
        **{key: value[eval_indices] for key, value in merged.items()},
        **metadata,
    )

    train_ids = merged["ids"][train_indices]
    eval_ids = merged["ids"][eval_indices]
    attempted = sum(
        int(summary["attempted"]) for summary in worker_summaries
    )
    validation = {
        "total_samples": int(len(merged["ids"])),
        "train_samples": int(len(train_ids)),
        "eval_samples": int(len(eval_ids)),
        "all_converged": bool(np.all(merged["converged"])),
        "all_finite": bool(
            np.all(np.isfinite(merged["u"]))
            and np.all(np.isfinite(merged["v"]))
        ),
        "max_residual_rms": float(np.max(merged["residual_rms"])),
        "median_residual_rms": float(
            np.median(merged["residual_rms"])
        ),
        "min_pattern_std": float(np.min(merged["pattern_std"])),
        "final_step_quantiles": np.quantile(
            merged["final_step"], [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
        ).astype(int).tolist(),
        "split_overlap": int(
            len(np.intersect1d(train_ids, eval_ids))
        ),
        "simulation_attempts": attempted,
        "simulation_acceptance_rate": cfg.total_samples / attempted,
        "A_range": [
            float(np.min(merged["A"])),
            float(np.max(merged["A"])),
        ],
        "B_range": [
            float(np.min(merged["B"])),
            float(np.max(merged["B"])),
        ],
        "d_range": [
            float(np.min(merged["d"])),
            float(np.max(merged["d"])),
        ],
        "outputs": {
            "params_csv": str(params_path),
            "convergence_report": str(report_path),
            "full_dataset": str(full_path),
            "train_dataset": str(train_path),
            "eval_dataset": str(eval_path),
        },
    }
    if validation["total_samples"] != cfg.total_samples:
        raise RuntimeError("Merged dataset has an incorrect sample count.")
    if validation["train_samples"] != cfg.train_samples:
        raise RuntimeError("Training split has an incorrect sample count.")
    if validation["eval_samples"] != cfg.eval_samples:
        raise RuntimeError("Evaluation split has an incorrect sample count.")
    if not validation["all_converged"] or not validation["all_finite"]:
        raise RuntimeError("A retained sample is invalid or unconverged.")
    if validation["max_residual_rms"] > cfg.convergence_rms:
        raise RuntimeError("A retained residual exceeds the tolerance.")
    if validation["split_overlap"] != 0:
        raise RuntimeError("Training and evaluation splits overlap.")

    summary_path = output_dir / "brusselator_generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": asdict(cfg), "validation": validation},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    validation["outputs"]["summary"] = str(summary_path)
    return validation


def parse_gpu_ids(raw: str) -> list[int]:
    gpu_ids = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not gpu_ids:
        raise argparse.ArgumentTypeError("At least one GPU ID is required.")
    return gpu_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate converged three-parameter Brusselator data."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("brusselator_data")
    )
    parser.add_argument("--gpus", type=parse_gpu_ids, default=[0, 1, 2, 3])
    parser.add_argument("--total-samples", type=int, default=20_000)
    parser.add_argument("--train-samples", type=int, default=16_000)
    parser.add_argument("--eval-samples", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.train_samples + args.eval_samples != args.total_samples:
        parser.error(
            "--train-samples + --eval-samples must equal --total-samples."
        )
    available_gpus = cp.cuda.runtime.getDeviceCount()
    if any(gpu < 0 or gpu >= available_gpus for gpu in args.gpus):
        parser.error(
            f"Requested GPUs {args.gpus}; only {available_gpus} are available."
        )

    cfg = GenerationConfig(
        total_samples=args.total_samples,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    base = cfg.total_samples // len(args.gpus)
    remainder = cfg.total_samples % len(args.gpus)
    targets = [
        base + (1 if index < remainder else 0)
        for index in range(len(args.gpus))
    ]
    worker_args = [
        (
            gpu_id,
            targets[index],
            str(shard_dir / f"brusselator_gpu{gpu_id}.npz"),
            cfg,
        )
        for index, gpu_id in enumerate(args.gpus)
    ]
    context = mp.get_context("spawn")
    with context.Pool(processes=len(args.gpus)) as pool:
        worker_summaries = pool.starmap(worker_generate, worker_args)
    validation = merge_split_validate(
        worker_summaries, cfg, args.output_dir
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    main()
