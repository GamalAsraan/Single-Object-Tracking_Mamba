#!/bin/bash
set -e

echo "Building Docker image..."
docker build -t trackingmamba-infer:latest .

echo -e "\nRunning import test..."
docker run --gpus all --rm trackingmamba-infer:latest python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
import mamba_ssm, causal_conv1d, timm, cv2
print("imports ok")
PY

echo -e "\nRunning infer_manifest --help..."
docker run --gpus all --rm \
  -v $(pwd)/models:/models:ro \
  trackingmamba-infer:latest \
  python -m runtime.infer_manifest --help
