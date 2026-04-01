# Use an official Python runtime as a parent image, slim version for smaller size
FROM python:3.14-slim

# Set working directory inside the container
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Define an entrypoint so the container runs as an executable CLI tool
ENTRYPOINT ["python", "-m", "rune"]
