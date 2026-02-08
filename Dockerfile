FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps:
# - chromium is required by Kaleido v1+ to export Plotly figures to images
# - build-essential is intentionally omitted for minimal image; psycopg2-binary is used
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        chromium \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Help Kaleido find a Chrome binary (it searches common names)
RUN if command -v chromium >/dev/null 2>&1; then \
      ln -sf "$(command -v chromium)" /usr/bin/google-chrome && \
      ln -sf "$(command -v chromium)" /usr/bin/google-chrome-stable; \
    fi

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Create runtime dirs (also mounted as volumes in docker-compose)
RUN mkdir -p /app/static/images /app/reports

EXPOSE 8000

# IMPORTANT: app.py stores per-session state in-memory (user_data_store).
# Multiple Gunicorn workers would each have their own memory, causing session_id lookups to fail.
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-8000} app:app"]
