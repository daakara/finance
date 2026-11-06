FROM python:3.11-slim

WORKDIR /app

# Python runtime security & buffering configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system build dependencies and runtime tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install python dependencies then purge compiler toolchain to reduce attack surface
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# NOTE: Production persistent volume is mounted to /root on Railway
# (analyst_dashboard.governance.storage.DEFAULT_PRODUCTION_VOLUME_ROOT).
# Retains standard entrypoint under root UID to satisfy physical device attestation.

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
