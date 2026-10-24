#!/usr/bin/env bash
# Adds the Google Drive OAuth client to the server: run in Cloud Shell after making the client.
#   bash deploy/gcp/drive.sh
set -euo pipefail
read -r -p "OAuth client ID: " CID
read -r -s -p "OAuth client secret: " CSEC; echo
read -r -p "Drive folder name [kinsey models 2026]: " FOLDER
printf 'GOOGLE_OAUTH_CLIENT_ID=%s\nGOOGLE_OAUTH_CLIENT_SECRET=%s\nJAGIMAGE_DRIVE_FOLDER=%s\n' "$CID" "$CSEC" "${FOLDER:-kinsey models 2026}" > ~/.jagstudio-drive
chmod 600 ~/.jagstudio-drive
bash "$(dirname "$0")/setup.sh"
