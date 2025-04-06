FROM python:3.10-slim

# Install system dependencies (e.g., build tools and libsndfile for audio I/O)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Runpod handler for MusicGen
COPY rp_handler.py .

CMD ["python3", "-u", "rp_handler.py"]
