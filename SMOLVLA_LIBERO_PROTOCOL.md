# SmolVLA LIBERO Reproduction Protocol

## Goal
Run `SmolVLA` evaluation on `LIBERO` from the local `LeRobot` checkout using local LIBERO assets and datasets already present on this machine.

## Local code paths
- LeRobot repo: `/root/workspace/andycho/IL/lerobot`
- LIBERO repo checkout: `/root/workspace/andycho/IL/VLA/LightVLA/LIBERO`
- LIBERO config: `/root/.libero/config.yaml`
- Eval wrapper: `/root/workspace/andycho/IL/lerobot/run_smolvla_libero_eval.sh`

## Python environment
Use the existing conda env:
- Python env: `/root/anaconda3/envs/LeAndy`
- Python version: `3.12`

This env was chosen because it already had the required runtime stack for the local LeRobot + LIBERO path:
- `torch`
- `transformers`
- `gymnasium`
- `robosuite`
- `bddl`

Additional dependency installed during setup:
- `num2words`

## Why this env was needed
- The default `python` in this workspace was `3.11`, but the local `IL/lerobot` checkout uses Python 3.12 syntax.
- The older installed `lerobot` package in the base env pointed to another checkout and was not appropriate for this run.

## LIBERO path setup
`LeRobot` uses `libero.libero.get_libero_path(...)` for benchmark assets.

`/root/.libero/config.yaml` was set to:

```yaml
benchmark_root: /root/workspace/andycho/IL/VLA/LightVLA/LIBERO/libero/libero
bddl_files: /root/workspace/andycho/IL/VLA/LightVLA/LIBERO/libero/libero/bddl_files
init_states: /root/workspace/andycho/IL/VLA/LightVLA/LIBERO/libero/libero/init_files
datasets: /root/workspace/andycho/IL/VLA/LightVLA/LIBERO/libero/datasets
assets: /root/workspace/andycho/IL/VLA/LightVLA/LIBERO/libero/libero/assets
```

## Dataset protocol
The four evaluation suites were found across two roots:
- `/root/datasets`
- `/root/dataset`

Symlinks created under:
- `/root/workspace/andycho/IL/VLA/LightVLA/LIBERO/libero/datasets`

Current suite mapping:
- `libero_spatial` -> `/root/dataset/libero_spatial`
- `libero_object` -> `/root/dataset/libero_object`
- `libero_goal` -> `/root/datasets/libero-goal/libero_goal`
- `libero_10` -> `/root/datasets/libero-100/libero_10`
- `libero_90` -> `/root/datasets/libero-100/libero_90`

## Local code patch applied
### 1. Policy import blocker fix
File:
- `/root/workspace/andycho/IL/lerobot/src/lerobot/policies/__init__.py`

Reason:
- `lerobot.policies` eagerly imported `GR00T`, which caused an unrelated import-time failure under the chosen runtime.
- `SmolVLA` evaluation does not depend on `GR00T`, so the import was made optional.

Effect:
- Local `LeRobot` import becomes usable for `SmolVLA` evaluation.

### 2. Video rendering control
File:
- `/root/workspace/andycho/IL/lerobot/src/lerobot/scripts/lerobot_eval.py`

Reason:
- The original code hardcoded `max_episodes_rendered=10` during eval.
- This adds significant rendering overhead during large benchmark runs.

Patch behavior:
- `max_episodes_rendered` now reads from env var:
- `LEROBOT_MAX_EPISODES_RENDERED`
- Default remains `10`
- Full reproduction run used `LEROBOT_MAX_EPISODES_RENDERED=0`

## Wrapper script
Wrapper created at:
- `/root/workspace/andycho/IL/lerobot/run_smolvla_libero_eval.sh`

Purpose:
- Always use the correct Python interpreter
- Export the local `LeRobot` and `LIBERO` paths via `PYTHONPATH`
- Export `LIBERO_CONFIG_PATH`
- Export `MUJOCO_GL=egl`

Equivalent environment setup:

```bash
export PYTHONPATH="/root/workspace/andycho/IL/lerobot/src:/root/workspace/andycho/IL/VLA/LightVLA/LIBERO"
export LIBERO_CONFIG_PATH="/root/.libero"
export MUJOCO_GL=egl
/root/anaconda3/envs/LeAndy/bin/python -m lerobot.scripts.lerobot_eval ...
```

## Smoke-test protocol used
Single-task smoke test:

```bash
cd /root/workspace/andycho/IL/lerobot

./run_smolvla_libero_eval.sh \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_10 \
  '--env.task_ids=[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.device=cuda \
  --output_dir=./outputs/eval/smolvla_libero_smoke
```

Observed result:
- rollout completed
- video written successfully when rendering was enabled
- example video path:
  `/root/workspace/andycho/IL/lerobot/outputs/eval/smolvla_libero_smoke/videos/libero_10_0/eval_episode_0.mp4`

## Full reproduction protocol
Full four-suite eval command:

```bash
cd /root/workspace/andycho/IL/lerobot

env LEROBOT_MAX_EPISODES_RENDERED=0 \
./run_smolvla_libero_eval.sh \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --policy.device=cuda \
  --output_dir=/root/workspace/andycho/IL/lerobot/outputs/eval/smolvla_libero_full_RUNID
```

## Detached background execution
Plain `nohup ... &` did not persist in this execution environment.

Working detached pattern:

```bash
setsid env LEROBOT_MAX_EPISODES_RENDERED=0 \
./run_smolvla_libero_eval.sh \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --policy.device=cuda \
  --output_dir=/root/workspace/andycho/IL/lerobot/outputs/eval/smolvla_libero_full_RUNID \
  > /root/workspace/andycho/IL/lerobot/nohup_smolvla_libero_full_RUNID.log 2>&1 < /dev/null &
```

## Current running full job
Started run:
- PID: `1267801`
- Log: `/root/workspace/andycho/IL/lerobot/nohup_smolvla_libero_full_20260406T092014Z.log`
- Output dir: `/root/workspace/andycho/IL/lerobot/outputs/eval/smolvla_libero_full_20260406T092014Z`

## Monitoring
Follow log:

```bash
tail -f /root/workspace/andycho/IL/lerobot/nohup_smolvla_libero_full_20260406T092014Z.log
```

Check process:

```bash
ps -fp 1267801
```

Stop job:

```bash
kill 1267801
```

## Notes on throughput
- `eval.batch_size` increases the number of parallel envs for the same task.
- `env.max_parallel_tasks` parallelizes across tasks.
- Single-episode latency remains dominated by sequential `policy -> env.step` rollout.
- Rendering videos adds significant overhead; disable it for full benchmark runs.
- Lowering LIBERO env render resolution may improve speed but can reduce visual fidelity before SmolVLA resizes inputs internally.
