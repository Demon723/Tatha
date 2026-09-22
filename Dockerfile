FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create checkpoints directory
RUN mkdir -p /app/checkpoints

# Expose ports
EXPOSE 8765  # WebSocket
EXPOSE 8766  # REST API

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('localhost',8765)); s.close()" || exit 1

# Run
CMD ["python", "tatha_realtime.py"]
