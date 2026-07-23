from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
import yaml

from runners.wovr.data_adapter import (
    chunk_frame_indices,
    read_action,
    read_video_frames,
)
from runners.wovr.train import load_dit_checkpoint, replace_action_mlps


def _load_episode_inputs(
    config: dict,
    record: dict,
    pose_p01: np.ndarray,
    pose_p99: np.ndarray,
):
    data_config = config["data"]
    raw_root = Path(data_config["raw_root"])
    frame_indices = np.arange(record["num_frames"], dtype=np.int64)
    action = read_action(
        raw_root / record["hdf5_path"],
        frame_indices,
        pose_p01,
        pose_p99,
    )
    first_image = read_video_frames(
        raw_root / record["video_path"],
        np.array([0], dtype=np.int64),
        data_config["width"],
        data_config["height"],
    )[0]
    return first_image, action


def _load_pipeline(config: dict, checkpoint_path: Path):
    model_config = config["model"]
    device = model_config["device"]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_config["torch_dtype"]]
    model_paths = [
        [str(Path(path)) for path in model_config["dit_paths"]],
        model_config["vae_path"],
    ]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(path=path, offload_device="cpu") for path in model_paths
        ],
    )
    replace_action_mlps(pipe.dit, config["action"]["dim"])
    load_dit_checkpoint(pipe.dit, checkpoint_path)
    pipe.dit.eval().to(device)
    pipe.vae.eval().to(device)
    return pipe


def infer(config: dict, pipe, first_image, action: np.ndarray, output_path: Path) -> Path:
    model_config = config["model"]
    inference_config = config["inference"]
    data_config = config["data"]
    device = model_config["device"]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_config["torch_dtype"]]
    num_chunks = math.ceil((len(action) - 1) / 8)

    generated_frames = []
    input_image4 = [first_image] * 4
    for chunk_id in range(num_chunks):
        indices = chunk_frame_indices(chunk_id, len(action))
        action_window = torch.from_numpy(action[indices]).to(device=device, dtype=dtype)
        chunk_frames = pipe(
            tiled=inference_config["tiled"],
            input_image=first_image,
            input_image4=input_image4,
            action=action_window,
            height=data_config["height"],
            width=data_config["width"],
            num_frames=13,
            num_inference_steps=inference_config["num_inference_steps"],
            cfg_scale=inference_config["cfg_scale"],
            idx=indices,
            bs_1=True,
        )[0]
        if chunk_id == 0:
            generated_frames.extend([chunk_frames[0], *chunk_frames[-8:]])
        else:
            generated_frames.extend(chunk_frames[5:])
        input_image4 = chunk_frames[-4:]

    generated_frames = generated_frames[: len(action)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(generated_frames, str(output_path), fps=data_config["fps"], quality=5)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WoVR inference on RoboTwin2.0")
    parser.add_argument(
        "--config", type=Path, default=Path("runners/wovr/configs/infer.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    val_manifest_path = Path(config["data"]["index_root"]) / "meta" / "val.jsonl"
    with val_manifest_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    with Path(config["action"]["statistics_path"]).open(
        "r", encoding="utf-8"
    ) as f:
        pose_statistics = json.load(f)["state_pose"]
    pose_p01 = np.asarray(pose_statistics["p01"], dtype=np.float32)
    pose_p99 = np.asarray(pose_statistics["p99"], dtype=np.float32)
    checkpoint_root = Path(config["model"]["checkpoint_root"])
    output_root = Path(config["inference"]["output_root"])
    for checkpoint_epoch in config["model"]["checkpoint_epochs"]:
        checkpoint_path = checkpoint_root / f"epoch-{checkpoint_epoch}.safetensors"
        pipe = _load_pipeline(config, checkpoint_path)
        checkpoint_output_root = output_root / f"epoch-{checkpoint_epoch}"
        for record in records[: config["inference"]["num_samples"]]:
            first_image, action = _load_episode_inputs(
                config,
                record,
                pose_p01,
                pose_p99,
            )
            output_path = (
                checkpoint_output_root
                / f"{record['task']}_episode{record['episode_id']}.mp4"
            )
            infer(config, pipe, first_image, action, output_path)
            print(f"Saved generated video to {output_path}")
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
