from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, Sampler


def chunk_frame_indices(chunk_id: int, episode_length: int) -> np.ndarray:
    if chunk_id == 0:
        indices = np.array(
            [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64
        )
    else:
        start = chunk_id * 8
        indices = np.array([0, *range(start - 3, start + 9)], dtype=np.int64)
    return np.clip(indices, 0, episode_length - 1)


def endposes_to_absolute_action(
    left_endpose: np.ndarray,
    right_endpose: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    pose_p01: np.ndarray,
    pose_p99: np.ndarray,
) -> np.ndarray:
    left_euler = Rotation.from_quat(
        left_endpose[:, 3:7], scalar_first=True
    ).as_euler("xyz", degrees=False)
    right_euler = Rotation.from_quat(
        right_endpose[:, 3:7], scalar_first=True
    ).as_euler("xyz", degrees=False)

    left_gripper_action = 1.0 - 2.0 * left_gripper
    right_gripper_action = 1.0 - 2.0 * right_gripper
    action = np.concatenate(
        [
            left_endpose[:, :3],
            left_euler,
            left_gripper_action[:, None],
            right_endpose[:, :3],
            right_euler,
            right_gripper_action[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    action[:, :6] = np.clip(
        2 * (action[:, :6] - pose_p01[:6])
        / (pose_p99[:6] - pose_p01[:6] + 1e-8)
        - 1.0,
        -1.0,
        1.0,
    )
    action[:, 7:13] = np.clip(
        2 * (action[:, 7:13] - pose_p01[7:13])
        / (pose_p99[7:13] - pose_p01[7:13] + 1e-8)
        - 1.0,
        -1.0,
        1.0,
    )
    return action


def read_video_frames(
    path: Path,
    frame_indices: np.ndarray,
    width: int,
    height: int,
) -> list[Image.Image]:
    reader = imageio.get_reader(path)
    frames_by_index = {}
    try:
        for frame_index in np.unique(frame_indices):
            frame = reader.get_data(frame_index)
            frames_by_index[frame_index] = (
                Image.fromarray(frame)
                .convert("RGB")
                .resize((width, height), Image.Resampling.BILINEAR)
            )
    finally:
        reader.close()
    return [frames_by_index[frame_index] for frame_index in frame_indices]


def read_action(
    path: Path,
    frame_indices: np.ndarray,
    pose_p01: np.ndarray,
    pose_p99: np.ndarray,
) -> np.ndarray:
    with h5py.File(path, "r") as f:
        left_endpose = f["/endpose/left_endpose"][:]
        right_endpose = f["/endpose/right_endpose"][:]
        left_gripper = f["/endpose/left_gripper"][:]
        right_gripper = f["/endpose/right_gripper"][:]

    action = endposes_to_absolute_action(
        left_endpose,
        right_endpose,
        left_gripper,
        right_gripper,
        pose_p01,
        pose_p99,
    )
    return action[frame_indices]


class RoboTwinDataset(Dataset):
    load_from_cache = False

    def __init__(
        self,
        raw_root: Path,
        manifest_path: Path,
        width: int,
        height: int,
        pose_statistics_path: Path,
    ) -> None:
        self.raw_root = raw_root
        with manifest_path.open("r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]
        self.width = width
        self.height = height
        with pose_statistics_path.open("r", encoding="utf-8") as f:
            pose_statistics = json.load(f)["state_pose"]
        self.pose_p01 = np.asarray(pose_statistics["p01"], dtype=np.float32)
        self.pose_p99 = np.asarray(pose_statistics["p99"], dtype=np.float32)

        self.samples = []
        self.episode_ranges = []
        for record in self.records:
            episode_start = len(self.samples)
            num_chunks = math.ceil((record["num_frames"] - 1) / 8)
            for chunk_id in range(num_chunks):
                self.samples.append((record, chunk_id))
            self.episode_ranges.append(range(episode_start, len(self.samples)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        record, chunk_id = self.samples[index]
        frame_indices = chunk_frame_indices(chunk_id, record["num_frames"])
        video = read_video_frames(
            self.raw_root / record["video_path"],
            frame_indices,
            self.width,
            self.height,
        )
        action = read_action(
            self.raw_root / record["hdf5_path"],
            frame_indices,
            self.pose_p01,
            self.pose_p99,
        )
        return {
            "action": torch.from_numpy(action),
            "episode_index": record["episode_id"],
            "idx": frame_indices,
            "reference_image": [video[0]],
            "task": record["task"],
            "video": video,
        }


class EpisodeSequentialSampler(Sampler[int]):
    def __init__(self, dataset: RoboTwinDataset, seed: int) -> None:
        self.episode_ranges = dataset.episode_ranges
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        episode_order = torch.randperm(
            len(self.episode_ranges), generator=generator
        ).tolist()
        for episode_index in episode_order:
            yield from self.episode_ranges[episode_index]

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.episode_ranges)


def build_index(config: dict) -> dict:
    data_config = config["data"]
    raw_root = Path(data_config["raw_root"])
    index_root = Path(data_config["index_root"])
    subset = data_config["subset"]

    hdf5_files = list(raw_root.glob(f"*/{subset}/data/episode*.hdf5"))

    records = []
    for hdf5_path in hdf5_files:
        episode_id = int(hdf5_path.stem.removeprefix("episode"))
        task_name = hdf5_path.parents[2].name
        with h5py.File(hdf5_path, "r") as f:
            num_frames = f["/endpose/left_endpose"].shape[0]
        video_path = hdf5_path.parent.parent / "video" / f"episode{episode_id}.mp4"
        records.append(
            {
                "episode_id": episode_id,
                "hdf5_path": hdf5_path.relative_to(raw_root).as_posix(),
                "num_frames": num_frames,
                "split": "train" if episode_id < 40 else "val",
                "task": task_name,
                "video_path": video_path.relative_to(raw_root).as_posix(),
            }
        )
    records.sort(key=lambda record: (record["task"], record["episode_id"]))

    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]

    meta_root = index_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    for split, split_records in (("train", train_records), ("val", val_records)):
        with (meta_root / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    info = {
        "camera": "head",
        "fps": data_config["fps"],
        "height": data_config["height"],
        "num_episodes": len(records),
        "num_frames_per_sample": data_config["num_frames"],
        "num_tasks": len({record["task"] for record in records}),
        "raw_root": str(raw_root),
        "split_counts": {"train": len(train_records), "val": len(val_records)},
        "split_rule": "episode0-39=train, episode40-49=val",
        "subset": subset,
        "width": data_config["width"],
    }
    (meta_root / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the RoboTwin index used by WoVR"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("runners/wovr/configs/train.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    info = build_index(config)
    print(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
