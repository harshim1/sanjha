#!/bin/sh
# Build the model from the latest data, seed the demo database, start the app.
#   ./run.sh           full rebuild (pulls data, refits, ~1 minute), then serve on :8000
#   ./run.sh serve     just serve what is already built
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
export PYTHONWARNINGS=ignore
if [ "$1" != "serve" ]; then
  .venv/bin/python -m app.build_model
  .venv/bin/python -m app.seed
  .venv/bin/python -m tests.test_live_loop
fi
exec .venv/bin/uvicorn app.main:app --port 8000
