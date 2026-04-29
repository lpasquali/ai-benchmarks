#!/bin/bash
# ==============================================================================
# RUNE — Local Development Image Builder
#
# Standardizes the build process for local RUNE images to ensure consistency
# between developer machines and the CI/CD pipeline (IEC 62443 4-1 ML4).
#
# Usage:
#   ./build-local.sh            # Build core 'rune' image
#   ./build-local.sh --all      # Build core, UI, Docs, and Audit images
#   ./build-local.sh --purge    # Stop containers and purge all Docker resources
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

# Handle purge first if requested
if [[ "$*" == *"--purge"* ]]; then
    echo -e "${BLUE}===> PURGING all Docker resources...${NC}"
    # Stop the current composition and remove volumes
    docker compose down -v --remove-orphans || true
    # Aggressive system-wide prune (images, containers, networks, volumes)
    docker system prune -af --volumes
    echo -e "${GREEN}===> Purge complete.${NC}"
    
    # If purge was the only argument, exit now.
    if [[ "$#" -eq 1 ]]; then
        exit 0
    fi
fi

# Auto-generate .env for local development
if [[ ! -f "$REPO_ROOT/.env" && -f "$REPO_ROOT/.env.example" ]]; then
    echo -e "${BLUE}===> Generating .env from template...${NC}"
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    RANDOM_PWD=$(openssl rand -hex 16)
    # Using sed -i with a backup extension for cross-platform compatibility (macOS/Linux)
    sed -i.bak "s/POSTGRES_PASSWORD=runesecret/POSTGRES_PASSWORD=${RANDOM_PWD}/" "$REPO_ROOT/.env"
    rm "$REPO_ROOT/.env.bak"
    echo -e "${GREEN}===> Generated .env with unique POSTGRES_PASSWORD${NC}"
fi

echo -e "${BLUE}===> Building RUNE core image...${NC}"
docker build -t "${REGISTRY}/rune:${TAG}" "$REPO_ROOT"

if [[ "$1" == "--all" ]]; then
    echo -e "${BLUE}===> Building RUNE-UI image...${NC}"
    docker build -t "${REGISTRY}/rune-ui:${TAG}" "$REPO_ROOT/../rune-ui"

    echo -e "${BLUE}===> Building RUNE-Docs image...${NC}"
    docker build -t "${REGISTRY}/rune-docs:${TAG}" "$REPO_ROOT/../rune-docs"

    echo -e "${BLUE}===> Building RUNE-Audit image...${NC}"
    docker build -t "${REGISTRY}/rune-audit:${TAG}" "$REPO_ROOT/../rune-audit"
fi

if [[ "$1" == "--push" || "$2" == "--push" ]]; then
    echo -e "${BLUE}===> Pushing images...${NC}"
    docker push "${REGISTRY}/rune:${TAG}"
    if [[ "$1" == "--all" ]]; then
        docker push "${REGISTRY}/rune-ui:${TAG}"
        docker push "${REGISTRY}/rune-docs:${TAG}"
        docker push "${REGISTRY}/rune-audit:${TAG}"
    fi
fi

echo -e "${BLUE}===> Updating Docker Compose services...${NC}"
# Use --remove-orphans to keep the environment clean
docker compose up -d --remove-orphans

echo -e "${GREEN}===> Build and deployment complete!${NC}"
echo -e "${GREEN}===> API: http://localhost:8080${NC}"
echo -e "${GREEN}===> UI:  http://localhost:3000${NC}"
echo -e "${GREEN}===> Docs: http://localhost:8000${NC}"
