#!/usr/bin/env bash
set -euo pipefail

APPLY="false"
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY="true" ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
: "${GOOGLE_CLOUD_REGION:?Set GOOGLE_CLOUD_REGION}"
: "${GOOGLE_HEALTHCARE_DATASET:?Set GOOGLE_HEALTHCARE_DATASET}"
: "${GOOGLE_DICOM_STORE:?Set GOOGLE_DICOM_STORE}"

command -v gcloud >/dev/null || {
  printf 'gcloud is required. Install Google Cloud CLI from the official repository.\n' >&2
  exit 1
}

commands=(
  "gcloud config set project $GOOGLE_CLOUD_PROJECT"
  "gcloud services enable healthcare.googleapis.com aiplatform.googleapis.com logging.googleapis.com"
  "gcloud healthcare datasets create $GOOGLE_HEALTHCARE_DATASET --location=$GOOGLE_CLOUD_REGION"
  "gcloud healthcare dicom-stores create $GOOGLE_DICOM_STORE --dataset=$GOOGLE_HEALTHCARE_DATASET --location=$GOOGLE_CLOUD_REGION"
)

printf 'Planned commands:\n'
printf '  %s\n' "${commands[@]}"
if [ "$APPLY" != "true" ]; then
  printf 'Plan only. Re-run with --apply after billing, region, IAM, and terms are approved.\n'
  exit 0
fi

for command in "${commands[@]}"; do
  bash -c "$command"
done

printf 'Healthcare dataset and anonymous DICOM Store are ready. Vertex MedGemma deployment remains a separately approved step.\n'
