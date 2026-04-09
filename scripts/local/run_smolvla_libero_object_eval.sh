#!/usr/bin/env bash
set -euo pipefail

# Quick runner for LIBERO object-only evals with optional n_action_steps sweep.
# Uses the existing environment wrapper: ./run_smolvla_libero_eval.sh

POLICY_PATH="HuggingFaceVLA/smolvla_libero"
N_EPISODES="5"
SEED="1000"
DEVICE="cuda"
USE_ASYNC_ENVS="false"
OUTPUT_ROOT="/root/workspace/andycho/IL/lerobot/outputs/eval"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
TASK_IDS=""
EPISODE_LENGTH=""
N_ACTION_STEPS="1"
SWEEP=""
DRY_RUN="false"
RENDER_EPISODES="0"

usage() {
  cat <<'EOF'
Usage:
  ./run_smolvla_libero_object_eval.sh [options]

Options:
  --policy-path <path>         Policy checkpoint path (default: HuggingFaceVLA/smolvla_libero)
  --n-episodes <int>           Episodes per task (default: 5)
  --seed <int>                 Eval seed (default: 1000)
  --device <cuda|cpu>          Policy device (default: cuda)
  --n-action-steps <int>       n_action_steps for single run (default: 1)
  --sweep <csv>                Sweep n_action_steps, e.g. 1,2,4 (overrides --n-action-steps)
  --use-async-envs <bool>      true/false (default: false)
  --task-ids <json-list>       Restrict task ids, e.g. [0,1,2]
  --episode-length <int>       Override env episode_length
  --output-root <dir>          Output root dir
  --run-id <id>                Run id suffix (default: UTC timestamp)
  --render-episodes <int>      Number of episodes to render to mp4 (default: 0)
  --dry-run                    Print commands only
  -h, --help                   Show this help

Examples:
  ./run_smolvla_libero_object_eval.sh --n-action-steps 1 --n-episodes 5
  ./run_smolvla_libero_object_eval.sh --sweep 1,2,4 --n-episodes 5 --seed 1000
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path)
      POLICY_PATH="$2"; shift 2 ;;
    --n-episodes)
      N_EPISODES="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --n-action-steps)
      N_ACTION_STEPS="$2"; shift 2 ;;
    --sweep)
      SWEEP="$2"; shift 2 ;;
    --use-async-envs)
      USE_ASYNC_ENVS="$2"; shift 2 ;;
    --task-ids)
      TASK_IDS="$2"; shift 2 ;;
    --episode-length)
      EPISODE_LENGTH="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --render-episodes)
      RENDER_EPISODES="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN="true"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -n "${SWEEP}" ]]; then
  IFS=',' read -r -a NAS_LIST <<< "${SWEEP}"
else
  NAS_LIST=("${N_ACTION_STEPS}")
fi

run_one() {
  local nas="$1"
  local out_dir="${OUTPUT_ROOT}/smolvla_libero_object_nas${nas}_${RUN_ID}"

  local cmd=(
    env "LEROBOT_MAX_EPISODES_RENDERED=${RENDER_EPISODES}"
    ./run_smolvla_libero_eval.sh
    "--policy.path=${POLICY_PATH}"
    "--env.type=libero"
    "--env.task=libero_object"
    "--eval.batch_size=1"
    "--eval.n_episodes=${N_EPISODES}"
    "--eval.use_async_envs=${USE_ASYNC_ENVS}"
    "--env.max_parallel_tasks=1"
    "--policy.device=${DEVICE}"
    "--policy.n_action_steps=${nas}"
    "--seed=${SEED}"
    "--output_dir=${out_dir}"
  )

  if [[ -n "${TASK_IDS}" ]]; then
    cmd+=("--env.task_ids=${TASK_IDS}")
  fi

  if [[ -n "${EPISODE_LENGTH}" ]]; then
    cmd+=("--env.episode_length=${EPISODE_LENGTH}")
  fi

  echo "============================================================"
  echo "Running libero_object eval"
  echo "n_action_steps=${nas} n_episodes=${N_EPISODES} seed=${SEED}"
  echo "output_dir=${out_dir}"
  echo "============================================================"

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

cd /root/workspace/andycho/IL/lerobot

for nas in "${NAS_LIST[@]}"; do
  run_one "${nas}"
done

echo "Done."
