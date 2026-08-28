#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="$ROOT/deploy/local-dicom"
DATA_ROOT="${MRI_STUDY_DATA_ROOT:-$HOME/mri-study-data}"
ORTHANC_ROOT="${MRI_STUDY_ORTHANC_ROOT:-$DATA_ROOT/services/orthanc}"
ENV_FILE="$ORTHANC_ROOT/stack.env"

mkdir -p "$ORTHANC_ROOT/secrets" "$ORTHANC_ROOT/postgres" "$ORTHANC_ROOT/storage"
chmod 700 "$ORTHANC_ROOT" "$ORTHANC_ROOT/secrets" "$ORTHANC_ROOT/postgres" "$ORTHANC_ROOT/storage"

if [ ! -s "$ENV_FILE" ]; then
  umask 077
  orthanc_password="$(openssl rand -hex 24)"
  postgres_password="$(openssl rand -hex 24)"
  printf '%s\n' "$postgres_password" > "$ORTHANC_ROOT/secrets/postgres_password"
  {
    printf 'MRI_STUDY_ORTHANC_ROOT=%s\n' "$ORTHANC_ROOT"
    printf 'MRI_STUDY_ORTHANC_USERNAME=mristudy\n'
    printf 'MRI_STUDY_ORTHANC_PASSWORD=%s\n' "$orthanc_password"
    printf 'MRI_STUDY_POSTGRES_PASSWORD=%s\n' "$postgres_password"
    printf 'MRI_STUDY_ORTHANC_BASIC_AUTH=%s\n' "$(printf 'mristudy:%s' "$orthanc_password" | base64 -w0)"
    # Keep this stack outside Docker's commonly allocated 172.16/12 space.
    # Every value remains configurable for hosts that already route 10.253.50/24.
    printf 'MRI_STUDY_DICOM_SUBNET=10.253.50.0/24\n'
    printf 'MRI_STUDY_POSTGRES_IP=10.253.50.10\n'
    printf 'MRI_STUDY_ORTHANC_IP=10.253.50.11\n'
    printf 'MRI_STUDY_OHIF_IP=10.253.50.12\n'
    printf 'MRI_STUDY_GATEWAY_IP=10.253.50.13\n'
  } > "$ENV_FILE"
fi

if ! grep -q '^MRI_STUDY_ORTHANC_BASIC_AUTH=' "$ENV_FILE"; then
  set -a
  . "$ENV_FILE"
  set +a
  printf 'MRI_STUDY_ORTHANC_BASIC_AUTH=%s\n' \
    "$(printf '%s:%s' "$MRI_STUDY_ORTHANC_USERNAME" "$MRI_STUDY_ORTHANC_PASSWORD" | base64 -w0)" \
    >> "$ENV_FILE"
fi

set -a
. "$ENV_FILE"
set +a

compose=(docker compose --env-file "$ENV_FILE" -f "$DEPLOY/compose.yaml")
case "$ACTION" in
  up) "${compose[@]}" up -d ;;
  down) "${compose[@]}" down ;;
  status) "${compose[@]}" ps ;;
  logs) "${compose[@]}" logs ;;
  env-path) printf '%s\n' "$ENV_FILE" ;;
  *) printf 'Usage: %s {up|down|status|logs|env-path}\n' "$0" >&2; exit 2 ;;
esac
