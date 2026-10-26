#!/bin/sh
# Starts Jag Studio in website mode. Railway gives the port in $PORT and the volume's path in
# $RAILWAY_VOLUME_MOUNT_PATH; photos, faces, keys and the Drive sign-in all live on that volume.
set -e
DATA="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"
if [ -z "$RAILWAY_VOLUME_MOUNT_PATH" ] && [ -n "$RAILWAY_ENVIRONMENT" ]; then
  echo "WARNING: no volume attached. Photos and settings are lost on every redeploy. Attach a volume at /data."
fi
if [ "${#JAGIMAGE_PASSWORD}" -lt 8 ]; then
  echo "ERROR: set the variable JAGIMAGE_PASSWORD (8 or more characters) in Railway > Variables, then redeploy."
  exit 1
fi
mkdir -p "$DATA/studio" "$DATA/jag-data"
export JAGIMAGE_DATA="$DATA/jag-data"
exec jagimage serve --cloud --host 0.0.0.0 --port "${PORT:-8080}" --out "$DATA/studio" --workers "${JAGIMAGE_WORKERS:-4}"
