#!/usr/bin/env bash
set -euo pipefail

VERSION="${SCW_CLI_VERSION:-2.61.0}"
ARCHIVE="scaleway-cli_${VERSION}_linux_amd64"
URL="https://github.com/scaleway/scaleway-cli/releases/download/v${VERSION}/${ARCHIVE}"
CHECKSUM_URL="https://github.com/scaleway/scaleway-cli/releases/download/v${VERSION}/SHA256SUMS"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fsSL "$URL" -o "$TMP/$ARCHIVE"
curl -fsSL "$CHECKSUM_URL" -o "$TMP/SHA256SUMS"
expected="$(grep "  $ARCHIVE$" "$TMP/SHA256SUMS" | cut -d' ' -f1)"
test -n "$expected"
actual="$(sha256sum "$TMP/$ARCHIVE" | cut -d' ' -f1)"
test "$actual" = "$expected"
sudo install -m 0755 "$TMP/$ARCHIVE" /usr/local/bin/scw
scw version
