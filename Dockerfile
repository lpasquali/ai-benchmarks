# Use an official Python runtime as a parent image, slim version for smaller size
FROM python:3.14-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Cloud tooling
# ---------------------------------------------------------------------------
# kubectl — required for HolmesGPT Kubernetes toolset
ARG KUBECTL_VERSION=v1.31.0
RUN curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    -o /usr/local/bin/kubectl \
    && chmod +x /usr/local/bin/kubectl \
    && kubectl version --client

# AWS CLI (adds `aws` command; boto3 is already installed via requirements.txt)
RUN pip install --no-cache-dir awscli

# Azure: azure-identity, azure-mgmt-* and azure-core are already installed
# via requirements.txt — they provide full programmatic access.
# azure-cli (~700 MB) is intentionally omitted to keep the image lean.
# Set AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID env vars,
# or use Azure Workload Identity on AKS (no credentials in the pod).

# GCP: google-cloud-aiplatform and google-auth are already installed via
# requirements.txt (pulled in by holmesgpt). Set GOOGLE_APPLICATION_CREDENTIALS
# to a mounted service-account JSON path, or use Workload Identity on GKE.

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
    RUNE_IDEMPOTENCY_KEY="" \
    VAST_API_KEY="" \
    \
    AWS_DEFAULT_REGION="" \
    AWS_ACCESS_KEY_ID="" \
    AWS_SECRET_ACCESS_KEY="" \
    AWS_SESSION_TOKEN="" \
    \
    AZURE_TENANT_ID="" \
    AZURE_SUBSCRIPTION_ID="" \
    AZURE_CLIENT_ID="" \
    AZURE_CLIENT_SECRET="" \
    \
    GOOGLE_CLOUD_PROJECT="" \
    GOOGLE_APPLICATION_CREDENTIALS=""

# Define an entrypoint so the container runs as an executable
# Use "serve" command by default for standalone API server mode
ENTRYPOINT ["python", "-m", "rune"]
CMD ["serve"]
