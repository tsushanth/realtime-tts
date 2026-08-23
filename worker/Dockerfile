# RunPod Serverless GPU worker. onnxruntime-gpu 1.17+ pulls its own CUDA 12 runtime libs
# via pip (nvidia-cublas-cu12, nvidia-cudnn-cu12, etc.) — RunPod's host already provides
# the GPU driver, so a full CUDA base image (10+GB, blew out even GitHub Actions' runner
# disk) isn't needed. A slim Python base is enough and keeps the image under ~1GB.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt onnxruntime-gpu

COPY synth.py handler.py .

# Model files are large (~340MB combined) and licensed for redistribution from the
# upstream release — fetched at build time so the image is self-contained.
RUN mkdir -p models && \
    curl -sL -o models/kokoro-v1.0.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx && \
    curl -sL -o models/voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

CMD ["python3", "-u", "handler.py"]
