FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend Code
COPY backend /app/backend

# Environment Variables
ENV PYTHONUNBUFFERED=1
ENV REDIS_URL=redis://redis:6379

# Expose API Port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
