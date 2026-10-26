# Jag Studio web app (Studio + Recolor + Agent + Google Drive) for Railway or any Docker host.
# Settings come from environment variables: JAGIMAGE_PASSWORD (required), and for Google Drive
# GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET. Keep photos on a volume mounted at /data.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY jagimage ./jagimage
RUN pip install -e ".[gemini]"
COPY web ./web
COPY deploy/railway/start.sh ./start.sh
EXPOSE 8080
CMD ["sh", "/app/start.sh"]
