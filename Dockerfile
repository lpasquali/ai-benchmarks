# Use an official Python runtime as a parent image, slim version for smaller size
FROM python:3.14-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (if any are needed, e.g., for certain Python packages)

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib\
    && rm -rf /var/lib/apt/lists/*  

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY rune rune_bench .

# Define an entrypoint so the container runs as an executable CLI tool
ENTRYPOINT ["python", "-m", "rune"]
