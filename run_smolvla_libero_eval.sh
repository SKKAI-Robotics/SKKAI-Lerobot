#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/workspace/andycho/IL/lerobot"
LIBERO_ROOT="/root/workspace/andycho/IL/VLA/LightVLA/LIBERO"
PYTHON_BIN="/root/anaconda3/envs/LeAndy/bin/python"

export PYTHONPATH="${REPO_ROOT}/src:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/root/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec "${PYTHON_BIN}" -m lerobot.scripts.lerobot_eval "$@"
