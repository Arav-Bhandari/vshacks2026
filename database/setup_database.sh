#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Install dependencies
echo "Installing dependencies..."
.venv/bin/pip install -q -r requirements.txt

# Fetch completed trials
echo "Fetching completed trials..."
cd backend
../.venv/bin/python -m app.database.fetch_trials

# Fetch remaining trials
echo "Fetching remaining trials..."
../.venv/bin/python -m app.database.fetch_trials 0 \
  "TERMINATED|WITHDRAWN|SUSPENDED|RECRUITING|ACTIVE_NOT_RECRUITING|ENROLLING_BY_INVITATION|NOT_YET_RECRUITING|UNKNOWN"

echo "Database setup complete."
