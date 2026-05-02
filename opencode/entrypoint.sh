#!/bin/sh
set -eu

WORKSPACE_DIR="${OPENCODE_WORKSPACE:-/workspace}"
WEB_HOST="${OPENCODE_WEB_HOST:-0.0.0.0}"
WEB_PORT="${OPENCODE_WEB_PORT:-4096}"
CONFIG_PATH="${OPENCODE_CONFIG:-$WORKSPACE_DIR/opencode.json}"

has_command() {
  command -v "$1" >/dev/null 2>&1
}

prepare_workspace() {
  mkdir -p "$WORKSPACE_DIR"
  cd "$WORKSPACE_DIR"
}

init_git_repo() {
  git init -b "${OPENCODE_GIT_DEFAULT_BRANCH:-main}" >/dev/null 2>&1 || \
    git init >/dev/null 2>&1 || true
}

ensure_initial_commit() {
  git rev-parse --verify HEAD >/dev/null 2>&1 && return 0
  git commit --allow-empty -m "${OPENCODE_INITIAL_COMMIT_MESSAGE:-chore: initial commit}" \
    >/dev/null 2>&1 || true
}

configure_git() {
  if ! has_command git; then
    echo "opencode web requires git inside the container image." >&2
    exit 1
  fi

  if [ "${OPENCODE_AUTO_INIT_GIT:-true}" = "true" ] && [ ! -d .git ]; then
    init_git_repo
  fi

  git config --global --add safe.directory "$WORKSPACE_DIR" >/dev/null 2>&1 || true
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0

  git config user.name >/dev/null 2>&1 || \
    git config --global user.name "${OPENCODE_GIT_USER_NAME:-OpenCode}" >/dev/null 2>&1 || true
  git config user.email >/dev/null 2>&1 || \
    git config --global user.email "${OPENCODE_GIT_USER_EMAIL:-opencode@local}" >/dev/null 2>&1 || true

  ensure_initial_commit
}

configure_model() {
  DEFAULT_MODEL="${LLM_MODEL:-}"
  [ -n "$DEFAULT_MODEL" ] || DEFAULT_MODEL="${LLM_MODEL_NAME:-}"

  OPENCODE_OPENAI_API_KEY="${OPENCODE_OPENAI_API_KEY:-${OPENAI_API_KEY:-local}}"
  OPENCODE_OPENAI_API_BASE="${OPENCODE_OPENAI_API_BASE:-${OPENAI_API_BASE:-${LLM_BASE_URL:-http://llama-cpp-server:11434/v1}}}"
  OPENCODE_PROVIDER_ID="${OPENCODE_PROVIDER_ID:-llamacpp}"
  OPENCODE_MODEL="${OPENCODE_MODEL:-${DEFAULT_MODEL:+${OPENCODE_PROVIDER_ID}/${DEFAULT_MODEL}}}"
  [ -n "$OPENCODE_MODEL" ] || OPENCODE_MODEL="${OPENCODE_PROVIDER_ID}/default"

  export CONFIG_PATH
  export OPENCODE_OPENAI_API_KEY
  export OPENCODE_OPENAI_API_BASE
  export OPENCODE_PROVIDER_ID
  export OPENCODE_MODEL
  export DEFAULT_MODEL

  python3 - <<'PY'
import json
import os

config_path = os.environ["CONFIG_PATH"]
provider_id = os.environ["OPENCODE_PROVIDER_ID"]
model = os.environ["OPENCODE_MODEL"]
model_name = os.environ.get("DEFAULT_MODEL") or model.split("/", 1)[-1]

config = {
    "$schema": "https://opencode.ai/config.json",
    "model": model,
    "provider": {
        provider_id: {
            "npm": "@ai-sdk/openai-compatible",
            "name": "llama.cpp",
            "options": {
                "apiKey": os.environ["OPENCODE_OPENAI_API_KEY"],
                "baseURL": os.environ["OPENCODE_OPENAI_API_BASE"],
            },
            "models": {
                model_name: {
                    "name": model_name,
                },
            },
        },
    },
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY

  export OPENCODE_CONFIG="$CONFIG_PATH"
}

run_opencode() {
  set -- opencode web --hostname "$WEB_HOST" --port "$WEB_PORT"

  if [ -n "${OPENCODE_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2086
    set -- "$@" $OPENCODE_EXTRA_ARGS
  fi

  exec "$@"
}

prepare_workspace
configure_git
configure_model

[ "$#" -eq 0 ] || exec "$@"
run_opencode
