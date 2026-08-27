#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -x "$ROOT/.venv/bin/hf" ]; then
  "$ROOT/.venv/bin/pip" install "huggingface-hub[cli]>=0.34,<1"
fi
"$ROOT/.venv/bin/hf" --help >/dev/null
printf 'Hugging Face CLI installed at %s/.venv/bin/hf\n' "$ROOT"
