FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Models (int8 only, ~70 MB): Apache-2.0 zipformer + silero VAD (MIT). Fetch with fetch_models.sh (public GitHub release assets).
COPY docker_models/ /models/
COPY server.py ./
ENV MODEL_DIR=/models PORT=8080 NUM_THREADS=1 MAX_CONNECTIONS=6
CMD ["python", "server.py"]
