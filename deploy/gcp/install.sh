#!/usr/bin/env bash
# Runs on the server (called by setup.sh): installs or updates Jag Studio and its HTTPS front door.
set -euo pipefail
: "${DOMAIN:?}"

id jag >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/jagstudio jag
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv caddy >/dev/null

mkdir -p /opt/jagstudio/app
rm -rf /opt/jagstudio/app.new && mkdir /opt/jagstudio/app.new
tar -xzf /tmp/jagstudio.tgz -C /opt/jagstudio/app.new
rm -rf /opt/jagstudio/app && mv /opt/jagstudio/app.new /opt/jagstudio/app
[ -d /opt/jagstudio/venv ] || python3 -m venv /opt/jagstudio/venv
/opt/jagstudio/venv/bin/pip install -q --upgrade pip
/opt/jagstudio/venv/bin/pip install -q -e "/opt/jagstudio/app[gemini]"

# settings: password (and Google Drive client, once you've made one), readable by root and jag only
{
  echo "JAGIMAGE_PASSWORD=$(cat /tmp/.jagstudio-password)"
  echo "JAGIMAGE_DATA=/var/lib/jagstudio/data"
  [ -f /tmp/.jagstudio-drive ] && cat /tmp/.jagstudio-drive
} > /etc/jagstudio.env
chown root:jag /etc/jagstudio.env && chmod 640 /etc/jagstudio.env
rm -f /tmp/.jagstudio-password /tmp/.jagstudio-drive /tmp/jagstudio.tgz
mkdir -p /var/lib/jagstudio/studio /var/lib/jagstudio/data && chown -R jag:jag /var/lib/jagstudio

cat > /etc/systemd/system/jagstudio.service <<UNIT
[Unit]
Description=Jag Studio
After=network-online.target

[Service]
User=jag
EnvironmentFile=/etc/jagstudio.env
ExecStart=/opt/jagstudio/venv/bin/jagimage serve --cloud --host 127.0.0.1 --port 8765 --out /var/lib/jagstudio/studio --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/caddy/Caddyfile <<CADDY
$DOMAIN {
	encode gzip
	reverse_proxy 127.0.0.1:8765
}
CADDY

systemctl daemon-reload
systemctl enable --now jagstudio >/dev/null
systemctl restart jagstudio caddy
sleep 3
systemctl is-active --quiet jagstudio && echo "Jag Studio is running."
