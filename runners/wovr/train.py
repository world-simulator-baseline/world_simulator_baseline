from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from diffsynth import load_state_dict
from diffsynth.trainers.utils import ModelLogger, launch_training_task
from torch import nn
from train_rlinf import WanTrainingModule
import yaml

from runners.wovr.data_adapter import EpisodeSequentialSampler, RoboTwinDataset


def replace_action_mlps(dit: nn.Module, action_dim: int) -> None:
    reference_parameter = next(dit.parameters())
    device = reference_parameter.device
    dtype = reference_parameter.dtype
    dim = dit.dim

    dit.action_mlp1 = nn.Sequential(
        nn.Linear(action_dim, dim),
        nn.GELU(),
        nn.Linear(dim, dim),
    ).to(device=device, dtype=dtype)
    dit.action_mlp2 = nn.Sequential(
        nn.Linear(4 * action_dim, 4 * dim),
        nn.SiLU(),
        nn.Linear(4 * dim, dim),
    ).to(device=device, dtype=dtype)
    for mlp in (dit.action_mlp1, dit.action_mlp2):
        for layer in (mlp[0], mlp[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


def load_dit_checkpoint(
    dit: nn.Module,
    checkpoint_path: Path,
) -> None:
    state_dict = load_state_dict(str(checkpoint_path))
    dit.load_state_dict(state_dict, strict=True)


def train(config: dict) -> None:
    data_config = config["data"]
    action_config = config["action"]
    model_config = config["model"]
    training_config = config["training"]

    train_dataset = RoboTwinDataset(
        raw_root=Path(data_config["raw_root"]),
        manifest_path=Path(data_config["index_root"]) / "meta" / "train.jsonl",
        width=data_config["width"],
        height=data_config["height"],
        pose_statistics_path=Path(action_config["statistics_path"]),
    )
    val_dataset = RoboTwinDataset(
        raw_root=Path(data_config["raw_root"]),
        manifest_path=Path(data_config["index_root"]) / "meta" / "val.jsonl",
        width=data_config["width"],
        height=data_config["height"],
        pose_statistics_path=Path(action_config["statistics_path"]),
    )
    train_sampler = EpisodeSequentialSampler(
        train_dataset,
        seed=training_config["episode_shuffle_seed"],
    )
    model_paths = [
        [str(Path(path)) for path in model_config["dit_paths"]],
        model_config["vae_path"],
    ]

    model = WanTrainingModule(
        model_paths=json.dumps(model_paths),
        trainable_models=model_config["trainable_models"],
        use_gradient_checkpointing=model_config["use_gradient_checkpointing"],
        use_gradient_checkpointing_offload=model_config[
            "use_gradient_checkpointing_offload"
        ],
        extra_inputs=model_config["extra_inputs"],
        max_timestep_boundary=model_config["max_timestep_boundary"],
        min_timestep_boundary=model_config["min_timestep_boundary"],
        context_noise_sigma=model_config["context_noise_sigma"],
        static_video_prob=model_config["static_video_prob"],
    )
    replace_action_mlps(model.pipe.dit, action_config["dim"])
    model.train()

    output_path = Path(training_config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    model_logger = ModelLogger(
        str(output_path),
        remove_prefix_in_ckpt=training_config["remove_prefix_in_ckpt"],
    )
    launch_args = SimpleNamespace(**training_config)
    print(
        f"Starting WoVR training with train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"resolution={data_config['width']}x{data_config['height']}"
    )
    launch_training_task(
        train_dataset,
        val_dataset,
        model,
        model_logger,
        train_sampler=train_sampler,
        args=launch_args,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WoVR on RoboTwin2.0")
    parser.add_argument(
        "--config", type=Path, default=Path("runners/wovr/configs/train.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == "__main__":
    main()
