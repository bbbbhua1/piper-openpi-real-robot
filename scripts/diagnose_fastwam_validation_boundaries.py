#!/usr/bin/env python3
"""Measure raw FastWAM joint-boundary violations on validation episode starts.

This script never opens a WebSocket or imports ROS.  It evaluates the same
checkpoint/runtime path used for Piper inference on the validation episode
starts, for one or more fixed diffusion seeds, and writes only a JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


_PIPER_ORDER_FROM_FASTWAM = np.asarray(
    [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13], dtype=np.intp
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--seeds",
        default="42,0,1,2",
        help="comma-separated fixed diffusion seeds (default: %(default)s)",
    )
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument(
        "--work-dir",
        default="/bh/zbh_self/tmp/fastwam-validation-boundary-diagnostic",
        help="temporary FastWAM loader work directory",
    )
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated integers") from exc
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def _entry_for_prediction(
    actions_fastwam: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    sample_index: int,
    seed: int,
) -> dict[str, Any]:
    actions_piper = actions_fastwam[:24, _PIPER_ORDER_FROM_FASTWAM]
    invalid = (actions_piper < lower) | (actions_piper > upper)
    result: dict[str, Any] = {
        "sample_index": sample_index,
        "seed": seed,
        "violating_values_first24": int(np.count_nonzero(invalid)),
        "left_j2_min": float(actions_piper[:, 1].min()),
        "left_j3_max": float(actions_piper[:, 2].max()),
        "left_gripper_min": float(actions_piper[:, 6].min()),
    }
    if np.any(invalid):
        timestep, piper_index = np.argwhere(invalid)[0].tolist()
        result["first_violation"] = {
            "timestep": int(timestep),
            "piper_index": int(piper_index),
            "value": float(actions_piper[timestep, piper_index]),
            "lower": float(lower[piper_index]),
            "upper": float(upper[piper_index]),
        }
    return result


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from piper_fastwam_policy_server import load_config

    config = load_config(args.config)
    fastwam_root = Path(str(config["fastwam_root"])).resolve()
    sys.path.insert(0, str(fastwam_root))
    sys.path.insert(0, str(fastwam_root / "src"))

    from fastwam.utils.misc import register_work_dir
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import torch

    from fastwam_piper_runtime import FastWAMPiperRuntime

    register_work_dir(args.work_dir)
    runtime = FastWAMPiperRuntime.from_config(config)
    with initialize_config_dir(version_base=None, config_dir=str(fastwam_root / "configs")):
        cfg = compose(
            config_name="sim_robotwin",
            overrides=[
                "task=agilex_cobotmagic2_puzzle_joint_3cam_384_1e-4_baidu32",
                f"data.val.pretrained_norm_stats={config['dataset_stats_path']}",
            ],
        )
    dataset = instantiate(cfg.data.val)
    episode_starts = [int(value) for value in dataset.lerobot_dataset.episode_data_index["from"]]

    lower = np.asarray(config["joint_lower"], dtype=np.float32)
    upper = np.asarray(config["joint_upper"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for sample_index in episode_starts:
        sample = dataset[sample_index]
        input_image = runtime._to_model_tensor(sample["video"][:, 0].unsqueeze(0).numpy())
        proprio = sample["proprio"][0].unsqueeze(0)
        for seed in seeds:
            with torch.no_grad():
                prediction = runtime.model.infer_action(
                    prompt=runtime.prompt,
                    input_image=input_image,
                    action_horizon=runtime.action_horizon,
                    num_video_frames=runtime.num_video_frames,
                    proprio=proprio,
                    negative_prompt=runtime.negative_prompt,
                    text_cfg_scale=runtime.text_cfg_scale,
                    num_inference_steps=runtime.num_inference_steps,
                    sigma_shift=runtime.sigma_shift,
                    seed=seed,
                    rand_device=runtime.rand_device,
                    tiled=runtime.tiled,
                )
            actions_fastwam = runtime._denormalize_action(prediction["action"])
            rows.append(
                _entry_for_prediction(
                    actions_fastwam,
                    lower,
                    upper,
                    sample_index=sample_index,
                    seed=seed,
                )
            )

    report = {
        "checkpoint": str(config["checkpoint_path"]),
        "dataset_stats_path": str(config["dataset_stats_path"]),
        "num_inference_steps": runtime.num_inference_steps,
        "episode_start_indices": episode_starts,
        "seeds": seeds,
        "execution_horizon_checked": 24,
        "runs": rows,
        "runs_with_violation": int(sum(row["violating_values_first24"] > 0 for row in rows)),
        "total_runs": len(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
