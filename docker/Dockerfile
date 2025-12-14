# Document Reconstruction Pipeline - Docker Image
# 
# Build:
#   docker build -t doc-recon -f docker/Dockerfile .
#
# Run CLI:
#   docker run -v $(pwd)/data:/data doc-recon python -m src.cli -i /data/input.pdf -o /data/output
#
# Run Web UI:
#   docker run -p 8080:8080 doc-recon

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    gcc \
    g++ \
    # Image processing
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # PDF processing
    poppler-utils \
    # OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # LaTeX (minimal for PDF generation)
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    # Utilities
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and cache
RUN mkdir -p /app/src/models /app/.cache

# Set permissions
RUN chmod +x /app/src/cli.py /app/src/app.py

# Expose port for web UI
EXPOSE 8080

# Default command: run the web UI
CMD ["streamlit", "run", "src/app.py", "--server.port=8080", "--server.address=0.0.0.0"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

