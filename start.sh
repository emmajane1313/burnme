#!/bin/bash
set -e

REPO_URL=${REPO_URL:-"https://github.com/emmajane1313/burnme.git"}
BRANCH=${BRANCH:-"main"}

cd /app

# Clone or pull latest code
if [ -d ".git" ]; then
  echo "Pulling latest changes from $BRANCH..."
  git fetch origin
  git reset --hard origin/$BRANCH
  git checkout 60c96de664391123a8b78e23256ae8f852657a47
else
  echo "Cloning repository from $REPO_URL..."
  git clone --branch $BRANCH $REPO_URL .
  git checkout 60c96de664391123a8b78e23256ae8f852657a47
fi

# Install Python dependencies
echo "Installing Python dependencies..."
uv sync --frozen

# Build frontend
echo "Building frontend..."
if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found. Installing nodejs and npm..."
  apt-get update
  apt-get install -y nodejs npm
fi
cd /app/frontend
npm install
npm run build

# Ensure assets are copied
mkdir -p /app/frontend/dist/assets/fonts /app/frontend/dist/assets/videos /app/frontend/dist/assets/images
cp -r /app/frontend/public/assets/fonts/* /app/frontend/dist/assets/fonts/ 2>/dev/null || true
cp -r /app/frontend/public/assets/videos/* /app/frontend/dist/assets/videos/ 2>/dev/null || true
cp -r /app/frontend/public/assets/images/* /app/frontend/dist/assets/images/ 2>/dev/null || true

cd /app

# Run the application
echo "Starting burnmewhileimhot..."
exec uv run burnmewhileimhot --host 0.0.0.0 --port 8000
