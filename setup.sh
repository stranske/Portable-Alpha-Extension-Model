#!/bin/bash
set -e
python3 -m venv .venv
source .venv/bin/activate
./scripts/setup_deps.sh
