#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m pip install --upgrade pip
pip install "pip-audit>=2.8,<3"
pip-audit -r requirements.txt
