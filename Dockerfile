FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpcap-dev \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-base.txt

COPY requirements-app.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-app.txt
# Copy Backend Code
COPY backend /app/backend

# Environment Variables
ENV PYTHONUNBUFFERED=1
ENV REDIS_URL=redis://redis:6379

# Expose API Port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
