#!/usr/bin/env bash
# Build and start the AlphaForge AI API via docker compose.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ] && [ -f .env.example ]; then
  echo "note: no .env found - copy .env.example to .env to add FRED/NewsAPI keys"
fi

docker compose -f docker/docker-compose.yml up --build -d
echo "API: http://127.0.0.1:8000/api/v1/health"
