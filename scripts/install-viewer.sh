#!/usr/bin/env bash
# Install the generated review bundle into a local viewer working directory.
#
# The viewer serves only from 127.0.0.1 and its study-specific files are kept
# out of Git. This copies the infrastructure and the generated bundle into a
# directory outside the repository, so no report, image or identifier is ever
# added to version control.
set -euo pipefail

STUDY="${1:-}"
TARGET="${2:-$HOME/atlas-irm}"

if [[ -z "$STUDY" ]]; then
  echo "usage: $0 <study-id> [target-directory]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${MRI_STUDY_DATA_ROOT:-$HOME/mri-study-data}"
BUNDLE="$DATA_ROOT/studies/$STUDY/reports/viewer"

if [[ ! -f "$BUNDLE/report-bundle.json" ]]; then
  echo "No viewer bundle for $STUDY. Run: ./bin/mristudy viewer-bundle --study $STUDY" >&2
  exit 1
fi

mkdir -p "$TARGET"
install -m 600 "$ROOT/report-webapp/server.py" "$TARGET/server.py"
install -m 600 "$ROOT/report-webapp/styles.css" "$TARGET/styles.css"
cp -a "$BUNDLE/." "$TARGET/"
chmod -R go-rwx "$TARGET"

echo "Viewer prepared in $TARGET"
echo "Start it with: python3 $TARGET/server.py"
echo "It listens only on 127.0.0.1:18043. Do not expose it with tailscale funnel."
