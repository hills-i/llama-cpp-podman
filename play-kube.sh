#!/bin/bash
set -eu

if (set -o pipefail) 2>/dev/null; then
  set -o pipefail
fi

# This script is used to run the project.
{
  cat config/model-config.yaml
  printf '\n---\n'
  cat config/postgresql-credentials.yaml
  printf '\n---\n'
  cat kube.yaml
} | podman play kube --network isolated -
