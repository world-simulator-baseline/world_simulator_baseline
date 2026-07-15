# World Simulator Baseline

A baseline world simulator for embodied AI research and evaluation.

## Quick Start

Clone the repository together with all baseline submodules:

```bash
git clone --recurse-submodules https://github.com/world-simulator-baseline/world_simulator_baseline.git
cd world_simulator_baseline
```

If you have already cloned the repository without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

The baseline repositories are available under [`third_party/`](third_party/).

## Baseline Runners

Raw robot datasets are converted once into the shared LeRobot format. Each baseline then uses its own data adapter to map LeRobot samples to the format required for training.

| Path | Purpose |
| --- | --- |
| `data_convert/robotwin.py` | Convert raw RoboTwin data to the shared LeRobot format |
| `runners/<baseline>/data_adapter.py` | Adapt LeRobot samples to the baseline-specific input format |
| `runners/<baseline>/train.py` | Load the training configuration, call the data adapter, and launch training |
| `runners/<baseline>/infer.py` | Load the inference configuration and launch inference |
| `runners/<baseline>/configs/train.yaml` | Training configuration |
| `runners/<baseline>/configs/infer.yaml` | Inference configuration |

> [!TIP]
> `train.py`, `infer.py`, and `data_adapter.py` are lightweight wrappers only; they do not patch or modify baseline source code.

### Data Workflow

```text
+-----------------------------------------------------------------+
| Data Layer 1: Raw RoboTwin                                      |
| <raw_robotwin_root>/                                            |
+--------------------------------+--------------------------------+
                                 |
                                 | Call
                                 v
+-----------------------------------------------------------------+
| python data_convert/robotwin.py                                 |
+--------------------------------+--------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
| Data Layer 2: LeRobot                                           |
| <lerobot_root>/                                                 |
| chmod -R a-w <lerobot_root>/                                    |
+---------+----------------------+----------------------+---------+
          |                      |                      |
          v                      v                      v
+-------------------+  +-------------------+  +-------------------+
| Call              |  | Call              |  | Call              |
| <baseline_a>/     |  | <baseline_b>/     |  | <baseline_n>/     |
| data_adapter.py   |  | data_adapter.py   |  | data_adapter.py   |
+---------+---------+  +---------+---------+  +---------+---------+
          |                      |                      |
          v                      v                      v
+-------------------+  +-------------------+  +-------------------+
| Data Layer 3: A   |  | Data Layer 3: B   |  | Data Layer 3: N   |
| <baseline_a_root>/|  | <baseline_b_root>/|  | <baseline_n_root>/|
+---------+---------+  +---------+---------+  +---------+---------+
          |                      |                      |
          v                      v                      v
+-------------------+  +-------------------+  +-------------------+
| Dataloader A      |  | Dataloader B      |  | Dataloader N      |
+-------------------+  +-------------------+  +-------------------+
```

## Development Workflow

```text
+--------------------------------------------------+
| Development branch                               |
| dev/<username>                                   |
| Role: Developer                                  |
+------------------------+-------------------------+
                         |
                         | Pull Request
                         | (Not enforced yet)
                         v
+--------------------------------------------------+
| Baseline fork: main / master                     |
| Role: Administrator reviews and merges           |
+------------------------+-------------------------+
                         |
                         | Update submodule commit
                         v
+--------------------------------------------------+
| Main repository                                  |
| Role: Administrator updates submodule reference  |
+--------------------------------------------------+
```

> [!TIP]
> Use Codex's `commit-style` skill to generate standardized, single-line commit messages in the `TYPE(SCOPE): SUBJECT` format. Keep them lowercase, imperative, under 50 characters, and without a trailing period.

Each directory under `third_party/` is an independent Git repository. Make and push baseline-specific changes from inside the corresponding submodule:

```bash
cd third_party/Ctrl-World
git switch dev/liuwenhao

# Edit and test the baseline code
git add <changed-files>
git commit -m "fix(core): describe the change"
git push
```

Once pull requests are enforced, open one from the development branch to the fork's `main` or `master` branch. After the pull request is merged, update the local default branch, then return to this repository and commit the updated submodule reference. This records which baseline commit the main repository should use; it does not duplicate the baseline code.

```bash
git switch main
git pull
cd ../..
git add third_party/Ctrl-World
git commit -m "chore(deps): update ctrl-world"
git push
```

Always push the submodule commit before updating the reference in this repository. Other contributors can then retrieve the exact versions with:

```bash
git pull
git submodule update --init --recursive
```

## Contributors

| Contributor | Responsibility |
| --- | --- |
| Wentao Tan | Main repository management, project standards, and baseline setup alignment |
| Yang Sun | Modify and maintain the WoVR baseline |
| Bowen Wang | Modify and maintain the GigaWorld-1 baseline |
| Wenhao Liu | Modify and maintain the Ctrl-World baseline |
| Zequn Wang | Modify and maintain the GE-Sim-V2 baseline |
| Zhe Li | Modify and maintain the WorldGym baseline |
| Xuebin Fang | Modify and maintain the Cosmos-Predict2.5 baseline |
