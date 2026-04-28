#!/bin/bash
# ==============================================================================
# RUNE — Local Development Image Builder
#
# Standardizes the build process for local RUNE images to ensure consistency
# between developer machines and the CI/CD pipeline (IEC 62443 4-1 ML4).
#
# Usage:
#   ./build-local.sh            # Build core 'rune' image
#   ./build-local.sh --all      # Build core, UI, and Docs images
#   ./build-local.sh --push     # Build and push to local registry (e.g. kind)
# ==============================================================================

set -e

# Repository Root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="${RUNE_LOCAL_REGISTRY:-ghcr.io/lpasquali}"
TAG="${RUNE_LOCAL_TAG:-latest}"

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${BLUE}===> Building RUNE core image...${NC}"
docker build -t "${REGISTRY}/rune:${TAG}" "$REPO_ROOT"

if [[ "$1" == "--all" ]]; then
    echo -e "${BLUE}===> Building RUNE-UI image...${NC}"
    docker build -t "${REGISTRY}/rune-ui:${TAG}" "$REPO_ROOT/../rune-ui"

    echo -e "${BLUE}===> Building RUNE-Docs image...${NC}"
    docker build -t "${REGISTRY}/rune-docs:${TAG}" "$REPO_ROOT/../rune-docs"
fi

if [[ "$1" == "--push" || "$2" == "--push" ]]; then
    echo -e "${BLUE}===> Pushing images...${NC}"
    docker push "${REGISTRY}/rune:${TAG}"
    if [[ "$1" == "--all" ]]; then
        docker push "${REGISTRY}/rune-ui:${TAG}"
        docker push "${REGISTRY}/rune-docs:${TAG}"
    fi
fi

echo -e "${GREEN}===> Build complete: ${REGISTRY}/rune:${TAG}${NC}"
