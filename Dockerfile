FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libturbojpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Use a virtualenv to avoid PEP 668 EXTERNALLY-MANAGED-ENVIRONMENT errors on Ubuntu 24.04
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip

# Install PyTorch compatible with CUDA 12.x
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy requirements and install
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy wheels and install in requested order
COPY wheels/ /tmp/wheels/
RUN pip install /tmp/wheels/causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl
RUN pip install /tmp/wheels/mamba_ssm-2.3.1-cp312-cp312-linux_x86_64.whl

# Verification RUN to ensure critical dependencies import properly
RUN python -c "\
import torch;\
import mamba_ssm;\
import causal_conv1d;\
import timm;\
import cv2;\
print('✅ All packages imported successfully!');\
print('Torch version:', torch.__version__);\
"

# Set up app directory
COPY app/ /app
WORKDIR /app
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "runtime.infer_manifest", "--help"]
