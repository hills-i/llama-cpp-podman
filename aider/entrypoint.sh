#!/bin/sh
set -eu

WORKSPACE_DIR="${AIDER_WORKSPACE:-/workspace}"
BROWSER_HOST="${AIDER_BROWSER_HOST:-0.0.0.0}"
BROWSER_PORT="${AIDER_BROWSER_PORT:-8501}"

has_command() {
  command -v "$1" >/dev/null 2>&1
}

prepare_workspace() {
  mkdir -p "$WORKSPACE_DIR"
  cd "$WORKSPACE_DIR"
}

init_git_repo() {
  git init -b "${AIDER_GIT_DEFAULT_BRANCH:-main}" >/dev/null 2>&1 || \
    git init >/dev/null 2>&1 || true
}

ensure_initial_commit() {
  git rev-parse --verify HEAD >/dev/null 2>&1 && return 0
  git commit --allow-empty -m "${AIDER_INITIAL_COMMIT_MESSAGE:-chore: initial commit}" \
    >/dev/null 2>&1 || true
}

configure_git() {
  if ! has_command git; then
    echo "aider --browser requires git inside the container image." >&2
    echo "Install git in the image or use a non-browser aider mode." >&2
    exit 1
  fi

  if [ "${AIDER_AUTO_INIT_GIT:-true}" = "true" ] && [ ! -d .git ]; then
    init_git_repo
  fi

  git config --global --add safe.directory "$WORKSPACE_DIR" >/dev/null 2>&1 || true
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0

  git config user.name >/dev/null 2>&1 || \
    git config --global user.name "${AIDER_GIT_USER_NAME:-Aider}" >/dev/null 2>&1 || true
  git config user.email >/dev/null 2>&1 || \
    git config --global user.email "${AIDER_GIT_USER_EMAIL:-aider@local}" >/dev/null 2>&1 || true

  ensure_initial_commit
}

configure_browser() {
  export STREAMLIT_SERVER_ADDRESS="$BROWSER_HOST"
  export STREAMLIT_SERVER_PORT="$BROWSER_PORT"
  export STREAMLIT_SERVER_HEADLESS=true
  export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
}

configure_model_env() {
  DEFAULT_MODEL="${LLM_MODEL:-}"
  [ -n "$DEFAULT_MODEL" ] || DEFAULT_MODEL="${LLM_MODEL_NAME:-}"

  AIDER_OPENAI_API_KEY="${AIDER_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}"
  AIDER_OPENAI_API_BASE="${AIDER_OPENAI_API_BASE:-${OPENAI_API_BASE:-${LLM_BASE_URL:-}}}"
  AIDER_MODEL="${AIDER_MODEL:-${DEFAULT_MODEL:+openai/${DEFAULT_MODEL}}}"

  [ -z "$AIDER_OPENAI_API_KEY" ] || export AIDER_OPENAI_API_KEY
  [ -z "$AIDER_OPENAI_API_BASE" ] || export AIDER_OPENAI_API_BASE
  [ -z "$AIDER_MODEL" ] || export AIDER_MODEL
}

run_aider() {
  set -- aider --browser --analytics-disable --no-check-update

  if [ -n "${AIDER_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2086
    set -- "$@" $AIDER_EXTRA_ARGS
  fi

  exec "$@"
}

prepare_workspace
configure_git
configure_browser
configure_model_env

[ "$#" -eq 0 ] || exec "$@"
run_aider
