#!/usr/bin/env bash
set -euo pipefail

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
: "${GOOGLE_CLOUD_REGION:?Set GOOGLE_CLOUD_REGION}"
: "${GOOGLE_HEALTHCARE_DATASET:?Set GOOGLE_HEALTHCARE_DATASET}"
: "${GOOGLE_DICOM_STORE:?Set GOOGLE_DICOM_STORE}"

APPLY="false"
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY="true" ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

printf 'Emergency cleanup target:\n'
printf '  project=%s\n  region=%s\n  dataset=%s\n  dicom_store=%s\n' \
  "$GOOGLE_CLOUD_PROJECT" "$GOOGLE_CLOUD_REGION" \
  "$GOOGLE_HEALTHCARE_DATASET" "$GOOGLE_DICOM_STORE"
printf 'This script only deletes the named DICOM Store. It does not infer other resources.\n'

if [ "$APPLY" != "true" ]; then
  printf 'Plan only. Re-run with --apply after verifying the exact target.\n'
  exit 0
fi

gcloud healthcare dicom-stores delete "$GOOGLE_DICOM_STORE" \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --location="$GOOGLE_CLOUD_REGION" \
  --dataset="$GOOGLE_HEALTHCARE_DATASET" \
  --quiet
