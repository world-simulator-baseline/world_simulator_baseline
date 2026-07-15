# RoboTwin raw -> LeRobot 2.1
"""Usage: python data_convert/robotwin_v21.py --origin PATH --output PATH [--subset NAME] [--fps N] [--robot-type TYPE] [--workers N]

Sibling of data_convert/robotwin.py (which targets LeRobot 3.0). The data read
from the raw HDF5 files and the field semantics are identical; only the on-disk
layout differs to follow the LeRobot 2.1 convention:

  - one parquet per episode        data/chunk-{chunk:03d}/episode_{idx:06d}.parquet
  - one mp4 per episode per camera videos/chunk-{chunk:03d}/{video_key}/episode_{idx:06d}.mp4
  - episode metadata as JSON lines meta/episodes.jsonl        (v3.0 used meta/episodes/*.parquet)
  - deduplicated task table        meta/tasks.jsonl
  - aggregate field statistics     meta/stats.json

Field mapping is unchanged: joint_action -> joint_delta, endpose -> eef_abs.
"""

import argparse, io, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py, numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

CHUNK_SIZE = 1000  # episodes per chunk directory (LeRobot 2.1 default)


def _image_hw(jpeg_bytes):
    """Return (height, width) of a JPEG frame by reading only its header."""
    with Image.open(io.BytesIO(bytes(jpeg_bytes))) as image:
        width, height = image.size
    return height, width


def read_episode(hdf5_path, instr_path):
    with h5py.File(hdf5_path, "r") as f:
        action = f["joint_action/vector"][:]  # (T, 14)
        left_gripper = f["endpose/left_gripper"][:].reshape(-1, 1)
        right_gripper = f["endpose/right_gripper"][:].reshape(-1, 1)
        state = np.concatenate([
            f["endpose/left_endpose"][:], left_gripper,
            f["endpose/right_endpose"][:], right_gripper,
        ], axis=1)  # (T, 16)
        num_frames = len(action)
        cam_keys = [k for k in f["observation"] if "rgb" in f[f"observation/{k}"]]
        images, cam_params = {}, {}
        for cam_key in cam_keys:
            name = cam_key.removesuffix("_camera")
            images[name] = [f[f"observation/{cam_key}/rgb"][i] for i in range(num_frames)]
            cam_params[name] = {
                "cam2world_gl": f[f"observation/{cam_key}/cam2world_gl"][:].reshape(num_frames, -1),
                "extrinsic_cv": f[f"observation/{cam_key}/extrinsic_cv"][:].reshape(num_frames, -1),
                "intrinsic_cv": f[f"observation/{cam_key}/intrinsic_cv"][:].reshape(num_frames, -1),
            }
        cam_raw_shapes = {}
        for cam_key in cam_keys:
            name = cam_key.removesuffix("_camera")
            cam_raw_shapes[name] = {
                "cam2world_gl": list(f[f"observation/{cam_key}/cam2world_gl"].shape[1:]),
                "extrinsic_cv": list(f[f"observation/{cam_key}/extrinsic_cv"].shape[1:]),
                "intrinsic_cv": list(f[f"observation/{cam_key}/intrinsic_cv"].shape[1:]),
            }
    with open(instr_path) as f:
        task_desc = json.load(f)["seen"][0]
    return action, state, images, cam_params, task_desc, cam_raw_shapes


def encode_video(jpegs, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps),
         "-i", "-", "-vf", "colorchannelmixer=rr=0:rb=1:br=1:bb=0",
         "-c:v", "libsvtav1", "-pix_fmt", "yuv420p", "-crf", "30",
         "-g", "2", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for jpeg in jpegs:
        process.stdin.write(bytes(jpeg))
    process.stdin.close()
    process.wait()


def _field_stats(data):
    return {
        "mean": data.mean(0).tolist(),
        "std": data.std(0).tolist(),
        "min": data.min(0).tolist(),
        "max": data.max(0).tolist(),
        "p01": np.percentile(data, 1, axis=0).tolist(),
        "p99": np.percentile(data, 99, axis=0).tolist(),
    }


def compute_stats(actions, states):
    all_actions = np.concatenate(actions)
    all_states = np.concatenate(states)
    return {"joint_delta": _field_stats(all_actions), "eef_abs": _field_stats(all_states)}


# --- Per-episode statistics for meta/episodes_stats.jsonl (LeRobot 2.1) ---
# These reproduce lerobot.datasets.compute_stats so the output loads with the stock
# library: numeric features are reduced over time; video features are sampled,
# normalized to [0, 1] and reduced to per-channel shape (3, 1, 1).


def _estimate_num_samples(n, min_num=100, max_num=10_000, power=0.75):
    if n < min_num:
        min_num = n
    return max(min_num, min(int(n ** power), max_num))


def _sample_indices(n):
    return np.round(np.linspace(0, n - 1, _estimate_num_samples(n))).astype(int).tolist()


def _feature_stats(array, axis, keepdims):
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def _numeric_stats(array):
    array = np.asarray(array)
    return _feature_stats(array, axis=0, keepdims=array.ndim == 1)


def _image_stats(jpegs):
    frames = []
    for i in _sample_indices(len(jpegs)):
        with Image.open(io.BytesIO(bytes(jpegs[i]))) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)  # (H, W, 3)
        # encode_video() swaps R<->B, so match that channel order for the stats too.
        chw = np.transpose(rgb[:, :, ::-1], (2, 0, 1))              # (3, H, W)
        frames.append(chw)
    stats = _feature_stats(np.stack(frames), axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def _serialize_stats(ep_stats):
    return {feature: {stat: value.tolist() for stat, value in per_stat.items()}
            for feature, per_stat in ep_stats.items()}


def _write_parquet(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_episode_outputs(episode_idx, action, state, images, cam_params, cameras,
                           global_offset, fps, task_index, output_dir):
    num_frames = len(action)
    chunk = episode_idx // CHUNK_SIZE

    # Standard LeRobot bookkeeping columns.
    timestamp = np.arange(num_frames, dtype=np.float32) / fps
    frame_index = np.arange(num_frames, dtype=np.int64)
    episode_index = np.full(num_frames, episode_idx, dtype=np.int64)
    index = np.arange(global_offset, global_offset + num_frames, dtype=np.int64)
    task_index_col = np.full(num_frames, task_index, dtype=np.int64)

    # Payload features first, then the bookkeeping columns.
    cols = {
        "joint_delta": [action[i].tolist() for i in range(num_frames)],
        "eef_abs": [state[i].tolist() for i in range(num_frames)],
    }
    for camera in cameras:
        for param_key, param_values in cam_params[camera].items():
            cols[f"observation.cameras.{camera}.{param_key}"] = [
                param_values[i].tolist() for i in range(num_frames)]
    cols["timestamp"] = timestamp
    cols["frame_index"] = frame_index
    cols["episode_index"] = episode_index
    cols["index"] = index
    cols["task_index"] = task_index_col

    _write_parquet(pa.table(cols),
                   output_dir / f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet")

    for camera in cameras:
        encode_video(images[camera],
                     output_dir / f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{episode_idx:06d}.mp4",
                     fps)

    # Per-episode stats for every feature (meta/episodes_stats.jsonl).
    ep_stats = {
        "joint_delta": _numeric_stats(action),
        "eef_abs": _numeric_stats(state),
        "timestamp": _numeric_stats(timestamp),
        "frame_index": _numeric_stats(frame_index),
        "episode_index": _numeric_stats(episode_index),
        "index": _numeric_stats(index),
        "task_index": _numeric_stats(task_index_col),
    }
    for camera in cameras:
        ep_stats[f"observation.images.{camera}"] = _image_stats(images[camera])
        for param_key, param_values in cam_params[camera].items():
            ep_stats[f"observation.cameras.{camera}.{param_key}"] = _numeric_stats(param_values)
    return ep_stats


def convert_task(task_dir, output_dir, subset, fps, robot_type, workers):
    data_dir = task_dir / subset / "data"
    instr_dir = task_dir / subset / "instructions"
    episodes = sorted(data_dir.glob("episode*.hdf5"),
                      key=lambda path: int(path.stem.removeprefix("episode")))

    # Phase 1: read all episodes in parallel
    episode_data = [None] * len(episodes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(read_episode, ep, instr_dir / f"{ep.stem}.json"): i
            for i, ep in enumerate(episodes)
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Reading", unit="ep", leave=False):
            episode_data[futures[future]] = future.result()

    # Phase 2: gather metadata + cumulative frame offsets (sequential, cheap).
    # The camera set, camera-parameter shapes and image dimensions are uniform across
    # episodes, so read them once from the first episode.
    _, _, first_images, _, _, cam_raw_shapes = episode_data[0]
    cameras = sorted(first_images.keys())
    img_shapes = {cam: [*_image_hw(first_images[cam][0]), 3] for cam in cameras}  # [H, W, C]

    all_actions, all_states, frame_counts, descriptions = [], [], [], []
    for action, state, _, _, task_desc, _ in episode_data:
        all_actions.append(action)
        all_states.append(state)
        frame_counts.append(len(action))
        descriptions.append(task_desc)

    global_offsets = [0]
    for fc in frame_counts[:-1]:
        global_offsets.append(global_offsets[-1] + fc)
    total_frames = sum(frame_counts)

    # Phase 3: write one parquet + one video per camera per episode, in parallel, and
    # collect per-episode stats. One task entry per episode, so task_index == episode_index.
    episode_stats = [None] * len(episode_data)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _write_episode_outputs, i, action, state, images, cam_params,
                cameras, global_offsets[i], fps, i, output_dir
            ): i
            for i, (action, state, images, cam_params, _, _) in enumerate(episode_data)
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Writing", unit="ep", leave=False):
            episode_stats[futures[future]] = future.result()

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # meta/tasks.jsonl: one entry per episode (task_index == episode_index)
    with open(meta_dir / "tasks.jsonl", "w") as tasks_file:
        for task_idx, task_desc in enumerate(descriptions):
            tasks_file.write(json.dumps({"task_index": task_idx, "task": task_desc}) + "\n")

    # meta/episodes.jsonl: one line per episode
    with open(meta_dir / "episodes.jsonl", "w") as episodes_file:
        for episode_idx, task_desc in enumerate(descriptions):
            episodes_file.write(json.dumps({
                "episode_index": episode_idx,
                "tasks": [task_desc],
                "length": frame_counts[episode_idx],
            }) + "\n")

    # meta/episodes_stats.jsonl: per-episode, per-feature stats (LeRobot 2.1 standard)
    with open(meta_dir / "episodes_stats.jsonl", "w") as stats_file:
        for episode_idx, ep_stats in enumerate(episode_stats):
            stats_file.write(json.dumps({
                "episode_index": episode_idx,
                "stats": _serialize_stats(ep_stats),
            }) + "\n")

    # meta/stats.json: aggregate stats over joint_delta / eef_abs (kept for convenience;
    # the v2.1 loader itself re-aggregates from episodes_stats.jsonl and ignores this file)
    stats = compute_stats(all_actions, all_states)
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    # meta/info.json
    camera_features = {}
    for camera in cameras:
        height, width, channels = img_shapes[camera]
        camera_features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": [height, width, channels],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(fps),
                "video.height": height, "video.width": width, "video.channels": channels,
                "video.codec": "av1", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "has_audio": False,
            },
        }
    camera_param_features = {}
    for camera in cameras:
        for param_key in ("cam2world_gl", "extrinsic_cv", "intrinsic_cv"):
            camera_param_features[f"observation.cameras.{camera}.{param_key}"] = {
                "dtype": "float32", "shape": cam_raw_shapes[camera][param_key], "names": None,
            }

    total_episodes = len(episodes)
    action_dim = all_actions[0].shape[1]
    state_dim = all_states[0].shape[1]
    info = {
        "codebase_version": "v2.1", "robot_type": robot_type, "fps": fps,
        "total_episodes": total_episodes, "total_frames": total_frames,
        "total_tasks": len(descriptions),
        "total_videos": total_episodes * len(cameras),
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "joint_delta": {"dtype": "float64", "shape": [action_dim], "names": None},
            "eef_abs": {"dtype": "float64", "shape": [state_dim], "names": None},
            **camera_features, **camera_param_features,
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
    return all_actions, all_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--subset", default="aloha-agilex_clean_50")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="aloha-agilex",
                        help="robot model name written to info.json (default: aloha-agilex)")
    parser.add_argument("--output", type=Path, required=True,
                        help="output directory")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="number of parallel workers (default: min(8, cpu_count))")
    args = parser.parse_args()

    origin_lerobot = args.output
    origin_lerobot.mkdir(parents=True, exist_ok=True)

    tasks = sorted(path for path in args.origin.iterdir()
                   if path.is_dir() and (path / args.subset / "data").exists())
    global_actions, global_states = [], []
    for task_dir in tqdm(tasks, desc="Tasks", unit="task"):
        tqdm.write(f"Converting {task_dir.name}...")
        output_dir = origin_lerobot / task_dir.name / args.subset
        actions, states = convert_task(
            task_dir, output_dir, args.subset, args.fps, args.robot_type, args.workers)
        global_actions.extend(actions)
        global_states.extend(states)

    global_stats_path = origin_lerobot / "stats.json"
    global_stats_path.write_text(json.dumps(compute_stats(global_actions, global_states), indent=2))
    tqdm.write(f"Done. Global stats -> {global_stats_path}")


if __name__ == "__main__":
    main()
