# RunPod Serverless GPU worker. onnxruntime-gpu does NOT pull CUDA runtime libs
# automatically via pip — that was a wrong assumption in an earlier version of this file
# and silently made the "GPU" worker fall back to CPU (confirmed: ort.get_available_providers()
# returned only CPUExecutionProvider, and measured RTF was *worse* than the CPU baseline).
# The nvidia-*-cu12 wheels in requirements.txt provide the actual .so files; LD_LIBRARY_PATH
# below is required for onnxruntime to find them. RunPod's host provides the GPU driver
# itself — a full CUDA base image (10+GB, blew out disk twice) still isn't needed.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt onnxruntime-gpu

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusolver/lib:${LD_LIBRARY_PATH}"

COPY synth.py handler.py .

# Model files are large (~340MB combined) and licensed for redistribution from the
# upstream release — fetched at build time so the image is self-contained.
RUN mkdir -p models && \
    curl -sL -o models/kokoro-v1.0.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx && \
    curl -sL -o models/voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

CMD ["python3", "-u", "handler.py"]
