FROM python:3.11-slim
RUN pip install --no-cache-dir "piper-tts==1.8.0" "numpy<2" scipy "fastapi==0.109.0" "uvicorn[standard]==0.27.0"
WORKDIR /app
COPY server.py .
# Model is NOT in git (63MB). Put full_ft.onnx and full_ft.onnx.json in ./models/ before building
# (see README.md for how to fetch them from the Modal volume).
COPY models/ /models/
ENV MODEL_PATH=/models/full_ft.onnx PORT=8080
CMD ["python", "server.py"]
