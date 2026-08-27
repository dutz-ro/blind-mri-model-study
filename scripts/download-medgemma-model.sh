#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION="${MEDGEMMA_REVISION:-91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b}"
DATA_ROOT="${MRI_STUDY_DATA_ROOT:-$HOME/mri-study-data}"
TARGET="$DATA_ROOT/services/scaleway/models/medgemma-1.5-4b-it/$REVISION"
MANIFEST="$TARGET/manifest.sha256"

mkdir -p "$TARGET"
chmod 700 "$DATA_ROOT/services/scaleway/models" "$DATA_ROOT/services/scaleway/models/medgemma-1.5-4b-it" "$TARGET"

"$ROOT/.venv/bin/python" - "$TARGET" "$REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

target, revision = sys.argv[1:]
snapshot_download(
    repo_id="google/medgemma-1.5-4b-it",
    revision=revision,
    local_dir=target,
    ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.tflite"],
)
PY

find "$TARGET" -type f ! -name manifest.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sed "s#  $TARGET/#  #" > "$MANIFEST"
chmod -R go-rwx "$TARGET"
(
  cd "$TARGET"
  sha256sum -c manifest.sha256 >/dev/null
)
printf 'Model snapshot: %s\n' "$TARGET"
printf 'Manifest SHA-256: %s\n' "$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
