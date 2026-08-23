# RunPod Serverless GPU worker. Three real GPU bugs found and fixed via live SSH
# debugging on an actual RunPod Pod before being baked in here — full trail in
# DECISIONS.md. Summary: (1) onnxruntime-gpu and plain onnxruntime collide at the same
# install path and kokoro-onnx pulls in plain onnxruntime transitively, silently
# disabling CUDA — fixed by installing onnxruntime-gpu first, then kokoro-onnx's other
# deps, then kokoro-onnx itself with --no-deps; (2) unpinned onnxruntime-gpu resolved to
# 1.29.0 which requires CUDA 13.x, but real CUDA-13 pip packages aren't published yet —
# pinned to 1.20.2 (CUDA 12.x + cuDNN 9.x, which has real packages); (3)
# nvidia-cuda-nvrtc-cu12 (libnvrtc.so.12, needed for CUDA JIT) was missing entirely.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir 'onnxruntime-gpu==1.20.2' espeakng-loader phonemizer && \
    pip install --no-cache-dir --no-deps kokoro-onnx==0.6.1 && \
    pip install --no-cache-dir soundfile numpy websockets runpod \
      nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 \
      nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/site-packages/nvidia/cusolver/lib:${LD_LIBRARY_PATH}"

COPY synth.py handler.py ./

# Model files are large (~340MB combined) and licensed for redistribution from the
# upstream release — fetched at build time so the image is self-contained.
RUN mkdir -p models && \
    curl -sL -o models/kokoro-v1.0.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx && \
    curl -sL -o models/voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

CMD ["python3", "-u", "handler.py"]
