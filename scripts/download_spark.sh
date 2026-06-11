#!/usr/bin/env bash
set -euo pipefail
mkdir -p assets
SPARK_VERSION="3.5.1"
SPARK_DIST="spark-${SPARK_VERSION}-bin-hadoop3"
TARBALL="${SPARK_DIST}.tgz"
URL="https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${TARBALL}"
DEST="assets/${TARBALL}"

if [ -f "${DEST}" ]; then
  echo "Tarball already exists at ${DEST}, skipping download."
  exit 0
fi

echo "Downloading ${URL} to ${DEST}..."
if command -v aria2c >/dev/null 2>&1; then
  aria2c -x 16 -s 16 -o "${DEST}" "${URL}"
else
  wget --progress=dot:giga -O "${DEST}" "${URL}"
fi

echo "Download complete: ${DEST}"
ls -lh "${DEST}"
