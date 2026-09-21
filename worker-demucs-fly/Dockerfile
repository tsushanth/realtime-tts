FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py ./
# Demucs pretrained weights (~80MB for htdemucs) are downloaded on first use into
# TORCH_HOME/hub (default /root/.cache/torch). Pre-warm the image so the first real request
# doesn't pay the download - comment out if you'd rather fetch at runtime (e.g. no network
# egress rules configured yet for this app).
RUN python -c "import os; os.environ.setdefault('AUTH_TOKEN','build'); from demucs.pretrained import get_model; get_model('htdemucs')"
ENV PORT=8080
CMD ["python", "server.py"]
