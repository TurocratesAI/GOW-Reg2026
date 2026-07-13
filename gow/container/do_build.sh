#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="reg2026_algorithm"

# Always re-stage first so the image reflects the CURRENT gow/ source + artifacts + weights (never a stale
# gowsrc/ or model/ snapshot). Set GOW_SKIP_STAGE=1 to reuse an existing staged tree.
if [ -z "${GOW_SKIP_STAGE:-}" ]; then
  bash "$SCRIPT_DIR/stage.sh"
fi

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR" 2>&1
