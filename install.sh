#!/usr/bin/env bash
set -euo pipefail

MODE="auto"
CHECK_ONLY="false"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode=docker|--mode=native) MODE="${1#*=}"; shift ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --check-only) CHECK_ONLY="true"; shift ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  auto|docker|native) ;;
  *) printf 'Mode must be auto, docker, or native.\n' >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${MRI_STUDY_DATA_ROOT:-$HOME/mri-study-data}"

command -v python3 >/dev/null || { printf 'python3 is required\n' >&2; exit 1; }
command -v git >/dev/null || { printf 'git is required\n' >&2; exit 1; }

AVAILABLE_KB="$(df -Pk "$HOME" | tr -s ' ' | cut -d' ' -f4 | tail -n 1)"
if [ "$AVAILABLE_KB" -lt 5242880 ]; then
  printf 'At least 5 GiB free space is required.\n' >&2
  exit 1
fi

if [ "$CHECK_ONLY" = "true" ]; then
  printf 'Checks passed. Data root: %s\n' "$DATA_ROOT"
  exit 0
fi

mkdir -p "$DATA_ROOT/studies"
chmod 700 "$DATA_ROOT" "$DATA_ROOT/studies"

if [ "$MODE" = "docker" ] || { [ "$MODE" = "auto" ] && command -v docker >/dev/null; }; then
  docker compose -f "$ROOT/compose.yaml" build
  printf 'Docker installation complete. Run: docker compose run --rm mristudy doctor\n'
  exit 0
fi

python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/pip" install -e "$ROOT[cloud,dicom-codecs,dev]"
"$ROOT/.venv/bin/mristudy" doctor
printf 'Native installation complete. Run: %s/.venv/bin/mristudy --help\n' "$ROOT"
