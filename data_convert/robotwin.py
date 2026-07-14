# RoboTwin raw -> LeRobot
# RoboTwin raw -> LeRobot 3.0
"""Usage: python data_convert/robotwin.py --origin PATH [--subset NAME] [--fps N] [--robot-type TYPE]"""

import argparse, json, subprocess
from pathlib import Path

import h5py, numpy as np
import pyarrow as pa, pyarrow.parquet as pq


def read_episode(hdf5_path, instr_path):
    with h5py.File(hdf5_path, "r") as f:
        action = f["joint_action/vector"][:]  # (T, 14)
        lg = f["endpose/left_gripper"][:].reshape(-1, 1)
        rg = f["endpose/right_gripper"][:].reshape(-1, 1)
        state = np.concatenate([
            f["endpose/left_endpose"][:], lg, f["endpose/right_endpose"][:], rg,
        ], axis=1)  # (T, 16)
        T = len(action)
        cam_keys = [k for k in f["observation"] if "rgb" in f[f"observation/{k}"]]
        images, cam_params = {}, {}
        for ck in cam_keys:
            name = ck.removesuffix("_camera")
            images[name] = [f[f"observation/{ck}/rgb"][i] for i in range(T)]
            cam_params[name] = {
                "cam2world_gl": f[f"observation/{ck}/cam2world_gl"][:].reshape(T, -1),
                "extrinsic_cv": f[f"observation/{ck}/extrinsic_cv"][:].reshape(T, -1),
                "intrinsic_cv": f[f"observation/{ck}/intrinsic_cv"][:].reshape(T, -1),
            }
        first_cam = cam_keys[0]
        img_shape = list(f[f"observation/{first_cam}/rgb"].shape[1:])  # e.g. [H, W, 3]
        cam_raw_shapes = {}
        for ck in cam_keys:
            name = ck.removesuffix("_camera")
            cam_raw_shapes[name] = {
                "cam2world_gl": list(f[f"observation/{ck}/cam2world_gl"].shape[1:]),
                "extrinsic_cv": list(f[f"observation/{ck}/extrinsic_cv"].shape[1:]),
                "intrinsic_cv": list(f[f"observation/{ck}/intrinsic_cv"].shape[1:]),
            }
    with open(instr_path) as fj:
        task_desc = json.load(fj)["seen"][0]
    return action, state, images, cam_params, task_desc, img_shape, cam_raw_shapes


def encode_video(jpegs, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for j in jpegs:
        p.stdin.write(bytes(j))
    p.stdin.close()
    p.wait()


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
    a, s = np.concatenate(actions), np.concatenate(states)
    return {"action": _field_stats(a), "observation.state": _field_stats(s)}


def _write_pq(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def convert_task(task_dir, out, subset, fps, robot_type):
    data_dir = task_dir / subset / "data"
    instr_dir = task_dir / subset / "instructions"
    eps = sorted(data_dir.glob("episode*.hdf5"),
                 key=lambda p: int(p.stem.removeprefix("episode")))

    descs = []
    for ep in eps:
        with open(instr_dir / f"{ep.stem}.json") as f:
            descs.append(json.load(f)["seen"][0])
    unique_tasks = list(dict.fromkeys(descs))
    task_map = {t: i for i, t in enumerate(unique_tasks)}

    all_a, all_s, gidx, cameras = [], [], 0, None
    img_shape, cam_raw_shapes = None, None

    for ei, ep in enumerate(eps):
        act, st, imgs, cam_p, _, ep_img_shape, ep_cam_shapes = read_episode(ep, instr_dir / f"{ep.stem}.json")
        T = len(act)
        if cameras is None:
            cameras = sorted(imgs.keys())
            img_shape = ep_img_shape
            cam_raw_shapes = ep_cam_shapes
        all_a.append(act)
        all_s.append(st)

        cols = {
            "index": np.arange(gidx, gidx + T, dtype=np.int64),
            "episode_index": np.full(T, ei, dtype=np.int64),
            "frame_index": np.arange(T, dtype=np.int64),
            "timestamp": np.arange(T, dtype=np.float64) / fps,
            "task_index": np.full(T, task_map[descs[ei]], dtype=np.int64),
            "action": [act[i].tolist() for i in range(T)],
            "observation.state": [st[i].tolist() for i in range(T)],
        }
        for c in cameras:
            for pk, pv in cam_p[c].items():
                cols[f"observation.cameras.{c}.{pk}"] = [pv[i].tolist() for i in range(T)]

        _write_pq(pa.table(cols), out / f"data/chunk-000/file-{ei:03d}.parquet")

        for c in cameras:
            encode_video(imgs[c], out / f"videos/observation.images.{c}/chunk-000/file-{ei:03d}.mp4", fps)

        _write_pq(pa.table({"episode_index": [ei], "tasks": [[descs[ei]]], "length": [T]}),
                  out / f"meta/episodes/chunk-000/file-{ei:03d}.parquet")
        gidx += T
        print(f"  ep {ei}: {T} frames")

    tasks_path = out / "meta/tasks.jsonl"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tasks_path, "w") as ft:
        for ti, td in enumerate(unique_tasks):
            ft.write(json.dumps({"task_index": ti, "task": td}) + "\n")

    stats = compute_stats(all_a, all_s)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta/stats.json").write_text(json.dumps(stats, indent=2))

    cam_f = {f"observation.images.{c}": {
        "dtype": "video", "shape": img_shape,
        "video_info": {"video.fps": fps, "video.codec": "h264",
                       "video.pix_fmt": "yuv420p",
                       "video.is_depth_map": False, "has_audio": False},
    } for c in cameras}
    cam_param_f = {}
    for c in cameras:
        for pk in ("cam2world_gl", "extrinsic_cv", "intrinsic_cv"):
            cam_param_f[f"observation.cameras.{c}.{pk}"] = {
                "dtype": "float32", "shape": cam_raw_shapes[c][pk],
            }

    action_dim = all_a[0].shape[1]
    state_dim = all_s[0].shape[1]
    (out / "meta/info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "robot_type": robot_type, "fps": fps,
        "total_episodes": len(eps), "total_frames": gidx,
        "features": {
            "action": {"dtype": "float64", "shape": [action_dim]},
            "observation.state": {"dtype": "float64", "shape": [state_dim]},
            **cam_f, **cam_param_f,
        },
        "data_path": "data/chunk-{chunk:03d}/file-{file:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk:03d}/file-{file:03d}.mp4",
        "episodes_path": "meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet",
    }, indent=2))
    return all_a, all_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", type=Path, required=True)
    ap.add_argument("--subset", default="aloha-agilex_clean_50")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--robot-type", default="aloha-agilex",
                    help="robot model name written to info.json (default: aloha-agilex)")
    args = ap.parse_args()

    tasks = sorted(p for p in args.origin.iterdir()
                   if p.is_dir() and (p / args.subset / "data").exists())
    ga, gs = [], []
    for t in tasks:
        print(f"Converting {t.name}...")
        a, s = convert_task(t, t / f"{args.subset}_lerobot", args.subset, args.fps, args.robot_type)
        ga.extend(a)
        gs.extend(s)

    (args.origin / "stats.json").write_text(json.dumps(compute_stats(ga, gs), indent=2))
    print(f"Done. Global stats -> {args.origin / 'stats.json'}")


if __name__ == "__main__":
    main()