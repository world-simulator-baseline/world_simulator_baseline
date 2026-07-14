# LeRobotDataset v3.0 Format

LeRobotDataset v3.0 standardizes dataset organization, temporal alignment, and metadata descriptions. It does not define robot-specific semantics such as joint ordering, physical units, or the meaning of each action dimension. These conventions must be documented and handled by the dataset converter and each baseline's data adapter.

A typical dataset has the following structure:

```text
dataset_root/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.jsonl
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    ├── observation.images.front/
    │   └── chunk-000/
    │       └── file-000.mp4
    └── observation.images.wrist/
        └── chunk-000/
            └── file-000.mp4
```

## Directory and Metadata Contents

- `data/` stores frame-level low-dimensional data, such as robot states, actions, and timestamps.
- `videos/` stores camera streams. Each camera is represented by a separate directory, such as `observation.images.front/` or `observation.images.wrist/`.
- `meta/info.json` defines dataset features, shapes, data types, FPS, and file path templates.
- `meta/stats.json` stores statistics for each field, including mean, standard deviation, minimum, and maximum values, for use in normalization.
- `meta/tasks.jsonl` stores natural-language task descriptions and their integer identifiers.
- `meta/episodes/` stores episode-level metadata, including episode lengths, associated tasks, data locations, and video locations.

## Robot-Specific Conventions

The LeRobotDataset v3.0 structure does not replace robot-specific schema definitions. Every converted dataset must explicitly document:

- joint ordering;
- units for robot states and actions;
- coordinate frames and rotation representations;
- the semantic meaning of each action dimension; and
- any normalization or preprocessing applied during conversion.


## Installation

https://huggingface.co/docs/lerobot/installation

## Keywords
- Mapping: joint_action -> joint_delta; endpose -> eef_abs
- eef_abs: absolute position(3) + absolute quaternion(4) + gripper(1) 
- gripper: all values are absolute
