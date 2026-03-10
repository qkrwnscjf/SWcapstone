FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    #libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY splits/ splits/

# Copy models if available
COPY models/ models/

# Expose ports
EXPOSE 8000

# Default command: serve
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
