# RoboTwin raw -> LeRobot 3.0
"""Usage: python data_convert/robotwin.py --origin PATH [--subset NAME] [--fps N] [--robot-type TYPE]"""

import argparse, json, subprocess
from pathlib import Path

import h5py, numpy as np
import pyarrow as pa, pyarrow.parquet as pq


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
        first_cam = cam_keys[0]
        img_shape = list(f[f"observation/{first_cam}/rgb"].shape[1:])  # e.g. [H, W, 3]
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
    return action, state, images, cam_params, task_desc, img_shape, cam_raw_shapes


def encode_video(jpegs, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(path)],
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


def _write_parquet(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def convert_task(task_dir, output_dir, subset, fps, robot_type):
    data_dir = task_dir / subset / "data"
    instr_dir = task_dir / subset / "instructions"
    episodes = sorted(data_dir.glob("episode*.hdf5"),
                      key=lambda path: int(path.stem.removeprefix("episode")))

    descriptions = []
    for episode_path in episodes:
        with open(instr_dir / f"{episode_path.stem}.json") as f:
            descriptions.append(json.load(f)["seen"][0])
    unique_tasks = list(dict.fromkeys(descriptions))
    task_map = {task: idx for idx, task in enumerate(unique_tasks)}

    all_actions, all_states, global_index, cameras = [], [], 0, None
    img_shape, cam_raw_shapes = None, None

    for episode_idx, episode_path in enumerate(episodes):
        action, state, images, cam_params, _, episode_img_shape, episode_cam_shapes = read_episode(
            episode_path, instr_dir / f"{episode_path.stem}.json")
        num_frames = len(action)
        if cameras is None:
            cameras = sorted(images.keys())
            img_shape = episode_img_shape
            cam_raw_shapes = episode_cam_shapes
        all_actions.append(action)
        all_states.append(state)

        cols = {
            "index": np.arange(global_index, global_index + num_frames, dtype=np.int64),
            "episode_index": np.full(num_frames, episode_idx, dtype=np.int64),
            "frame_index": np.arange(num_frames, dtype=np.int64),
            "timestamp": np.arange(num_frames, dtype=np.float64) / fps,
            "task_index": np.full(num_frames, task_map[descriptions[episode_idx]], dtype=np.int64),
            "joint_delta": [action[i].tolist() for i in range(num_frames)],
            "eef_abs": [state[i].tolist() for i in range(num_frames)],
        }
        for camera in cameras:
            for param_key, param_values in cam_params[camera].items():
                cols[f"observation.cameras.{camera}.{param_key}"] = [
                    param_values[i].tolist() for i in range(num_frames)]

        data_parquet = output_dir / f"data/chunk-000/file-{episode_idx:03d}.parquet"
        _write_parquet(pa.table(cols), data_parquet)

        for camera in cameras:
            video_path = output_dir / f"videos/observation.images.{camera}/chunk-000/file-{episode_idx:03d}.mp4"
            encode_video(images[camera], video_path, fps)

        episode_parquet = output_dir / f"meta/episodes/chunk-000/file-{episode_idx:03d}.parquet"
        _write_parquet(
            pa.table({"episode_index": [episode_idx], "tasks": [[descriptions[episode_idx]]], "length": [num_frames]}),
            episode_parquet)
        global_index += num_frames
        print(f"episode {episode_idx}: {num_frames} frames")

    tasks_path = output_dir / "meta/tasks.jsonl"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tasks_path, "w") as tasks_file:
        for task_idx, task_desc in enumerate(unique_tasks):
            tasks_file.write(json.dumps({"task_index": task_idx, "task": task_desc}) + "\n")

    stats = compute_stats(all_actions, all_states)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    stats_path = meta_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    camera_features = {
        f"observation.images.{camera}": {
            "dtype": "video", "shape": img_shape,
            "video_info": {
                "video.fps": fps, "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "has_audio": False,
            },
        } for camera in cameras
    }
    camera_param_features = {}
    for camera in cameras:
        for param_key in ("cam2world_gl", "extrinsic_cv", "intrinsic_cv"):
            camera_param_features[f"observation.cameras.{camera}.{param_key}"] = {
                "dtype": "float32", "shape": cam_raw_shapes[camera][param_key],
            }

    action_dim = all_actions[0].shape[1]
    state_dim = all_states[0].shape[1]
    info_path = meta_dir / "info.json"
    info_path.write_text(json.dumps({
        "codebase_version": "v3.0", "robot_type": robot_type, "fps": fps,
        "total_episodes": len(episodes), "total_frames": global_index,
        "features": {
            "joint_delta": {"dtype": "float64", "shape": [action_dim]},
            "eef_abs": {"dtype": "float64", "shape": [state_dim]},
            **camera_features, **camera_param_features,
        },
        "data_path": "data/chunk-{chunk:03d}/file-{file:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk:03d}/file-{file:03d}.mp4",
        "episodes_path": "meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet",
    }, indent=2))
    return all_actions, all_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--subset", default="aloha-agilex_clean_50")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="aloha-agilex",
                        help="robot model name written to info.json (default: aloha-agilex)")
    args = parser.parse_args()

    tasks = sorted(path for path in args.origin.iterdir()
                   if path.is_dir() and (path / args.subset / "data").exists())
    global_actions, global_states = [], []
    for task_dir in tasks:
        print(f"Converting {task_dir.name}...")
        actions, states = convert_task(
            task_dir, task_dir / f"{args.subset}_lerobot", args.subset, args.fps, args.robot_type)
        global_actions.extend(actions)
        global_states.extend(states)

    global_stats_path = args.origin / "stats.json"
    global_stats_path.write_text(json.dumps(compute_stats(global_actions, global_states), indent=2))
    print(f"Done. Global stats -> {global_stats_path}")


if __name__ == "__main__":
    main()
