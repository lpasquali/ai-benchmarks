# ==============================================================================
# Stage 1 — builder
# Compiler toolchain, git and kubectl download are confined to this stage and
# do NOT appear in the final image.
# IEC 62443 4-1 ML4 SAD-3: minimise software attack surface
# ==============================================================================
FROM python:3.14-slim AS builder

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    build-essential libjpeg-dev zlib1g-dev git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# kubectl — fetch binary here so the final image needs no curl/wget/git
ARG KUBECTL_VERSION=v1.31.0
RUN curl -fsSL \
    "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    -o /usr/local/bin/kubectl \
 && chmod +x /usr/local/bin/kubectl

# Isolated virtualenv — only this directory is copied to the final stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --prefer-binary --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir awscli

# ==============================================================================
# Stage 2 — final runtime
# No compiler, no git, non-root user, OS packages fully patched at build time.
# IEC 62443 4-1 ML4: principle of least privilege (SM-5, SAD-3, DM-4)
# ==============================================================================
FROM python:3.14-slim AS final

# Apply all available OS security patches at image build time (IEC 62443 DM-4)
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Non-root, no-login runtime user (IEC 62443 4-1 ML4 least-privilege)
RUN groupadd -r rune && useradd -r -g rune -u 1000 -d /app -s /sbin/nologin rune

WORKDIR /app

# Bring in pre-compiled Python packages and kubectl — nothing else from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/bin/kubectl /usr/local/bin/kubectl

# Application source
COPY rune rune_bench ./
RUN chown -R rune:rune /app

ENV PATH="/opt/venv/bin:$PATH" \
    RUNE_BACKEND=local \
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
    RUNE_KUBECONFIG=/app/.kube/config \
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

USER rune
EXPOSE 8080

# Define an entrypoint so the container runs as an executable
# Use "serve" command by default for standalone API server mode
ENTRYPOINT ["python", "-m", "rune"]
CMD ["serve"]
