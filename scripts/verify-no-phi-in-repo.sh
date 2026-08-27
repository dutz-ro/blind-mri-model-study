#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bad="$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null | while read -r path; do
  case "$path" in
    *.dcm|*.dicom|*.ima|*.nii|*.nii.gz|*.nrrd|*.mha|*.mhd|*.raw|*.pem|*.key|*.env|*service-account*.json)
      printf '%s\n' "$path"
      ;;
  esac
done)"

if [ -n "$bad" ]; then
  printf 'Blocked protected or secret-like files:\n%s\n' "$bad" >&2
  exit 1
fi
printf 'Staged file policy passed. This does not replace a manual PHI review.\n'
