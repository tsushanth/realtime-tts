# RunPod Serverless GPU worker. Base image provides CUDA + Python; onnxruntime-gpu picks
# up the GPU automatically when available (falls back to CPU otherwise).
FROM runpod/base:0.6.2-cuda12.1.0

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
