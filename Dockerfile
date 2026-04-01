# Use an official Python runtime as a parent image, slim version for smaller size
FROM python:3.14-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (if any are needed, e.g., for certain Python packages)

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev build-essential\
    && rm -rf /var/lib/apt/lists/*  

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY rune rune_bench .

# Environment variables with defaults for RUNE configuration
# Backend and API configuration
ENV RUNE_BACKEND=local \
    RUNE_API_HOST=0.0.0.0 \
    RUNE_API_PORT=8080 \
    RUNE_API_BASE_URL=http://localhost:8080 \
    RUNE_API_TOKEN="" \
    RUNE_API_TENANT=default \
    RUNE_DEBUG=false \
    \
    RUNE_VASTAI=false \
    RUNE_VASTAI_TEMPLATE=c166c11f035d3a97871a23bd32ca6aba \
    RUNE_VASTAI_MIN_DPH=2.3 \
    RUNE_VASTAI_MAX_DPH=3.0 \
    RUNE_VASTAI_RELIABILITY=0.99 \
    RUNE_VASTAI_STOP_INSTANCE=false \
    \
    RUNE_OLLAMA_URL=http://localhost:11434 \
    RUNE_OLLAMA_WARMUP=true \
    RUNE_OLLAMA_WARMUP_TIMEOUT=300 \
    \
    RUNE_QUESTION="" \
    RUNE_MODEL=llama3.1:8b \
    RUNE_KUBECONFIG=/root/.kube/config \
    RUNE_IDEMPOTENCY_KEY=""

# Define an entrypoint so the container runs as an executable
# Use "serve" command by default for standalone API server mode
ENTRYPOINT ["python", "-m", "rune"]
CMD ["serve"]
