#!/bin/bash
# RUNE — Local Build Script for Docker Compose
# Builds all core components from source in ~/Devel/

set -e

# Repository directory
BASE_DIR="/home/luca/Devel"
RUNE_DIR="${BASE_DIR}/rune"

cd "${RUNE_DIR}"

echo "--- 🛠️ Building RUNE locally ---"

# 1. Ensure docker-compose.override.yml is present (it should be, we just created it)
if [ ! -f "docker-compose.override.yml" ]; then
    echo "⚠️ Warning: docker-compose.override.yml missing. Creating it..."
    cat > docker-compose.override.yml <<EOF
services:
  rune-api:
    build:
      context: .
      dockerfile: Dockerfile
  rune-ui:
    build:
      context: ../rune-ui
      dockerfile: Dockerfile
  rune-docs:
    build:
      context: ../rune-docs
      dockerfile: Dockerfile
EOF
fi

# 2. Build the services
echo "--- 🚀 Running docker compose build ---"
docker compose build --parallel

echo ""
echo "--- ✅ Build complete! ---"
echo "You can now start the stack with:"
echo "  docker compose up -d"
echo ""
echo "To view logs:"
echo "  docker compose logs -f"
