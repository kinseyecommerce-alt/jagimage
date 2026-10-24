#!/usr/bin/env bash
# Jag Studio on Google Cloud: run this in Google Cloud Shell from the unzipped jagimage folder.
#   bash deploy/gcp/setup.sh
# It creates one small server (Compute Engine) in Mumbai with a fixed address, copies Jag Studio
# onto it, and starts it behind HTTPS at https://<address>.sslip.io. Safe to run again to update.
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-south1}"
ZONE="${ZONE:-asia-south1-a}"
VM="${VM:-jag-studio}"
MACHINE="${MACHINE:-e2-medium}"   # 2 vCPU (shared), 4 GB; e2-standard-2 for faster recolouring
DISK_GB="${DISK_GB:-100}"

[ -n "$PROJECT" ] || { echo "Pick your project first: gcloud config set project YOUR-PROJECT-ID"; exit 1; }
[ -f pyproject.toml ] && [ -d jagimage ] || { echo "Run this from the unzipped jagimage folder"; exit 1; }
echo "Project $PROJECT, server $VM ($MACHINE, ${DISK_GB} GB) in $ZONE"

gcloud services enable compute.googleapis.com drive.googleapis.com --project "$PROJECT"

if ! gcloud compute addresses describe "$VM-ip" --region "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud compute addresses create "$VM-ip" --region "$REGION" --project "$PROJECT"
fi
IP=$(gcloud compute addresses describe "$VM-ip" --region "$REGION" --project "$PROJECT" --format='value(address)')
DOMAIN="${IP//./-}.sslip.io"

if ! gcloud compute firewall-rules describe jag-studio-web --project "$PROJECT" >/dev/null 2>&1; then
  gcloud compute firewall-rules create jag-studio-web --project "$PROJECT" --allow tcp:80,tcp:443 --target-tags jag-studio
fi

if ! gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE" \
    --image-family debian-12 --image-project debian-cloud --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
    --address "$IP" --tags jag-studio
  echo "Waiting for the server to boot..."; sleep 40
fi

if [ ! -f ~/.jagstudio-password ]; then
  read -r -s -p "Choose a password for the Jag Studio website (8+ characters): " PW; echo
  [ ${#PW} -ge 8 ] || { echo "Too short"; exit 1; }
  printf '%s' "$PW" > ~/.jagstudio-password; chmod 600 ~/.jagstudio-password
fi

echo "Copying Jag Studio to the server..."
tar --exclude=./output --exclude=./.git --exclude='*/__pycache__' -czf /tmp/jagstudio.tgz .
gcloud compute scp /tmp/jagstudio.tgz ~/.jagstudio-password deploy/gcp/install.sh "$VM":/tmp/ --zone "$ZONE" --project "$PROJECT"
[ -f ~/.jagstudio-drive ] && gcloud compute scp ~/.jagstudio-drive "$VM":/tmp/ --zone "$ZONE" --project "$PROJECT"
gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --command "sudo DOMAIN=$DOMAIN bash /tmp/install.sh"

echo
echo "Jag Studio is at:  https://$DOMAIN"
echo "Google Drive redirect address (for the OAuth client): https://$DOMAIN/drive/callback"
