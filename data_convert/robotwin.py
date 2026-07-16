# RoboTwin raw -> LeRobot 2.1
"""Usage: python data_convert/robotwin.py --origin PATH --output PATH [--subset NAME] [--fps N] [--robot-type TYPE] [--workers N]

Converts RoboTwin raw HDF5 data to LeRobot 2.1 on-disk layout:

  - one parquet per episode        data/chunk-{chunk:03d}/episode_{idx:06d}.parquet
  - one mp4 per episode per camera videos/chunk-{chunk:03d}/{video_key}/episode_{idx:06d}.mp4
  - episode metadata as JSON lines meta/episodes.jsonl
  - deduplicated task table        meta/tasks.jsonl
  - aggregate field statistics     meta/stats.json

Field mapping: joint_action -> joint_abs, endpose -> eef_abs.
"""

import argparse
import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

CHUNK_SIZE = 1000
CAMERA_PARAM_KEYS = ("cam2world_gl", "extrinsic_cv", "intrinsic_cv")


@dataclass
class EpisodeData:
    action: np.ndarray
    state: np.ndarray
    images: dict
    cam_params: dict
    task_desc: str


def read_episode(hdf5_path, instr_path):
    with h5py.File(hdf5_path, "r") as f:
        action = f["joint_action/vector"][:]

        left_endpose = f["endpose/left_endpose"][:]
        left_gripper = f["endpose/left_gripper"][:].reshape(-1, 1)
        right_endpose = f["endpose/right_endpose"][:]
        right_gripper = f["endpose/right_gripper"][:].reshape(-1, 1)
        state = np.concatenate(
            [left_endpose, left_gripper, right_endpose, right_gripper], axis=1,
        )

        num_frames = len(action)
        obs_group = f["observation"]
        cam_keys = [k for k in obs_group if "rgb" in obs_group[k]]

        images = {}
        cam_params = {}
        for cam_key in cam_keys:
            name = cam_key.removesuffix("_camera")
            cam_group = obs_group[cam_key]
            images[name] = [bytes(cam_group["rgb"][i]) for i in range(num_frames)]
            cam_params[name] = {k: cam_group[k][:] for k in CAMERA_PARAM_KEYS}

    with open(instr_path) as f:
        task_desc = json.load(f)["seen"][0]

    return EpisodeData(action, state, images, cam_params, task_desc)


def encode_video(jpegs, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps),
            "-i", "-", "-vf", "colorchannelmixer=rr=0:rb=1:br=1:bb=0",
            "-c:v", "libsvtav1", "-pix_fmt", "yuv420p", "-crf", "30",
            "-g", "2", str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for jpeg in jpegs:
        process.stdin.write(jpeg)
    process.stdin.close()
    process.wait()


def _field_stats(data, axis=0, keepdims=False):
    return {
        "min": np.min(data, axis=axis, keepdims=keepdims),
        "max": np.max(data, axis=axis, keepdims=keepdims),
        "mean": np.mean(data, axis=axis, keepdims=keepdims),
        "std": np.std(data, axis=axis, keepdims=keepdims),
        "count": np.array([len(data)]),
    }


def _numeric_stats(array):
    return _field_stats(array, axis=0, keepdims=(array.ndim == 1))


def compute_aggregate_stats(actions, states):
    result = {}
    for name, data in [("joint_abs", np.concatenate(actions)),
                        ("eef_abs", np.concatenate(states))]:
        stats = _field_stats(data)
        stats["p01"] = np.percentile(data, 1, axis=0)
        stats["p99"] = np.percentile(data, 99, axis=0)
        result[name] = {k: v.tolist() for k, v in stats.items() if k != "count"}
    return result


def _sample_indices(n, min_num=100, max_num=10_000, power=0.75):
    min_num = min(min_num, n)
    num_samples = max(min_num, min(int(n ** power), max_num))
    indices = np.linspace(0, n - 1, num_samples)
    return np.round(indices).astype(int).tolist()


def _image_stats(jpegs):
    indices = _sample_indices(len(jpegs))
    frames = []
    for i in indices:
        with Image.open(io.BytesIO(jpegs[i])) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        # Source JPEGs are BGR; swap to RGB to match encode_video output.
        rgb = rgb[:, :, ::-1]
        chw = np.transpose(rgb, (2, 0, 1))
        frames.append(chw)

    stats = _field_stats(np.stack(frames), axis=(0, 2, 3), keepdims=True)
    for key in ("min", "max", "mean", "std"):
        stats[key] = np.squeeze(stats[key] / 255.0, axis=0)
    return stats


def _write_episode(episode_idx, ep, cameras, global_offset, fps, output_dir):
    num_frames = len(ep.action)
    chunk = episode_idx // CHUNK_SIZE

    timestamp = np.arange(num_frames, dtype=np.float32) / fps
    frame_index = np.arange(num_frames, dtype=np.int64)
    episode_index = np.full(num_frames, episode_idx, dtype=np.int64)
    index = np.arange(global_offset, global_offset + num_frames, dtype=np.int64)
    task_index = np.zeros(num_frames, dtype=np.int64)

    cols = {
        "joint_abs": ep.action.tolist(),
        "eef_abs": ep.state.tolist(),
    }
    for camera in cameras:
        for param_key, param_values in ep.cam_params[camera].items():
            rows = param_values.shape[1]
            inner = param_values.shape[2]
            matrix_type = pa.list_(pa.list_(pa.float32(), inner), rows)
            col_name = f"observation.cameras.{camera}.{param_key}"
            cols[col_name] = pa.array(param_values.tolist(), type=matrix_type)
    cols["timestamp"] = timestamp
    cols["frame_index"] = frame_index
    cols["episode_index"] = episode_index
    cols["index"] = index
    cols["task_index"] = task_index

    parquet_path = output_dir / f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), parquet_path)

    for camera in cameras:
        video_path = (
            output_dir
            / f"videos/chunk-{chunk:03d}/observation.images.{camera}"
            / f"episode_{episode_idx:06d}.mp4"
        )
        encode_video(ep.images[camera], video_path, fps)

    ep_stats = {
        "joint_abs": _numeric_stats(ep.action),
        "eef_abs": _numeric_stats(ep.state),
        "timestamp": _numeric_stats(timestamp),
        "frame_index": _numeric_stats(frame_index),
        "episode_index": _numeric_stats(episode_index),
        "index": _numeric_stats(index),
        "task_index": _numeric_stats(task_index),
    }
    for camera in cameras:
        ep_stats[f"observation.images.{camera}"] = _image_stats(ep.images[camera])
        for param_key, param_values in ep.cam_params[camera].items():
            ep_stats[f"observation.cameras.{camera}.{param_key}"] = _numeric_stats(
                param_values,
            )
    return ep_stats


def _write_meta_files(output_dir, task_name, episodes, cameras, img_shapes,
                      episode_stats, aggregate_stats, fps, robot_type,frame_counts):
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = len(episodes)
    total_frames = sum(frame_counts)
    action_dim = episodes[0].action.shape[1]
    state_dim = episodes[0].state.shape[1]

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_name}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i, ep in enumerate(episodes):
            record = {
                "episode_index": i,
                "tasks": [task_name],
                "instruction": ep.task_desc,
                "length": frame_counts[i],
            }
            f.write(json.dumps(record) + "\n")

    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for i, ep_stats in enumerate(episode_stats):
            record = {
                "episode_index": i,
                "stats": {
                    feature: {stat: value.tolist() for stat, value in per_stat.items()}
                    for feature, per_stat in ep_stats.items()
                },
            }
            f.write(json.dumps(record) + "\n")

    (meta_dir / "stats.json").write_text(json.dumps(aggregate_stats, indent=2))

    features = {
        "joint_abs": {"dtype": "float64", "shape": [action_dim], "names": None},
        "eef_abs": {"dtype": "float64", "shape": [state_dim], "names": None},
    }
    for camera in cameras:
        h, w, c = img_shapes[camera]
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [h, w, c],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.height": h,
                "video.width": w,
                "video.channels": c,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        for param_key in CAMERA_PARAM_KEYS:
            features[f"observation.cameras.{camera}.{param_key}"] = {
                "dtype": "float32",
                "shape": list(episodes[0].cam_params[camera][param_key].shape[1:]),
                "names": None,
            }
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(cameras),
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))


def convert_task(task_dir, output_dir, subset, fps, robot_type, workers):
    data_dir = task_dir / subset / "data"
    instr_dir = task_dir / subset / "instructions"
    hdf5_files = sorted(
        data_dir.glob("episode*.hdf5"),
        key=lambda p: int(p.stem.removeprefix("episode")),
    )

    # Phase 1: read all episodes in parallel.
    episodes = [None] * len(hdf5_files)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, ep_path in enumerate(hdf5_files):
            instr_path = instr_dir / f"{ep_path.stem}.json"
            future = pool.submit(read_episode, ep_path, instr_path)
            futures[future] = i
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Reading", unit="ep", leave=False):
            episodes[futures[future]] = future.result()

    # Phase 2: gather metadata from first episode (sequential, cheap).
    first_ep = episodes[0]
    cameras = sorted(first_ep.images.keys())
    img_shapes = {}
    for cam in cameras:
        with Image.open(io.BytesIO(first_ep.images[cam][0])) as img:
            img_shapes[cam] = [img.height, img.width, 3]

    frame_counts = [len(ep.action) for ep in episodes]
    global_offsets = [0]
    for fc in frame_counts[:-1]:
        global_offsets.append(global_offsets[-1] + fc)

    # Phase 3: write parquet + videos in parallel.
    episode_stats = [None] * len(episodes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, ep in enumerate(episodes):
            future = pool.submit(
                _write_episode, i, ep, cameras, global_offsets[i], fps, output_dir,
            )
            futures[future] = i
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Writing", unit="ep", leave=False):
            episode_stats[futures[future]] = future.result()

    # Phase 4: write meta files.
    all_actions = [ep.action for ep in episodes]
    all_states = [ep.state for ep in episodes]
    aggregate_stats = compute_aggregate_stats(all_actions, all_states)
    _write_meta_files(
        output_dir, task_dir.name, episodes, cameras, img_shapes,
        episode_stats, aggregate_stats, fps, robot_type, frame_counts,
    )

    return all_actions, all_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--subset", default="aloha-agilex_clean_50")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="aloha-agilex")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(
        path for path in args.origin.iterdir()
        if path.is_dir() and (path / args.subset / "data").exists()
    )
    global_actions = []
    global_states = []
    for task_dir in tqdm(task_dirs, desc="Tasks", unit="task"):
        tqdm.write(f"Converting {task_dir.name}...")
        output_dir = args.output / task_dir.name / args.subset
        actions, states = convert_task(
            task_dir, output_dir, args.subset, args.fps, args.robot_type, args.workers,
        )
        global_actions.extend(actions)
        global_states.extend(states)

    global_stats = compute_aggregate_stats(global_actions, global_states)
    global_stats_path = args.output / "stats.json"
    global_stats_path.write_text(json.dumps(global_stats, indent=2))
    tqdm.write(f"Done. Global stats -> {global_stats_path}")


if __name__ == "__main__":
    main()
