# syntax=docker/dockerfile:1

# --- Builder stage ---
FROM python:3.11-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu --retries 10 --timeout 120 --prefix=/install -r requirements.txt

# --- Final stage (minimal image) ---
FROM python:3.11-slim
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run the app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 