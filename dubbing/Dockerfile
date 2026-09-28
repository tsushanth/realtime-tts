FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg for retiming + any other system deps
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy source
COPY dubbing/ ./dubbing/
COPY voices/ ./voices/

# No pip installs needed — pure stdlib + subprocess(ffmpeg)

EXPOSE 8090

CMD ["python3", "-m", "dubbing.job_server", "--host", "0.0.0.0", "--port", "8090"]
