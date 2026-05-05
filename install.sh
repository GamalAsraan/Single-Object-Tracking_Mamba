#!/bin/bash

echo "========================================="
echo "  TrackingMamba - Kaggle Setup Script"
echo "========================================="

echo ""
echo ">>> Step 1: Install Mamba wheels"
WHEEL_DIR="/kaggle/input/mamba-code/TrackingMamba/Mamba_wheel"
pip install ${WHEEL_DIR}/causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl
pip install ${WHEEL_DIR}/mamba_ssm-2.3.1-cp312-cp312-linux_x86_64.whl

echo ""
echo ">>> Step 2: Install core dependencies"
pip install easydict pycocotools jpeg4py lmdb colorama thop \
    tikzplotlib tensorboardX wandb einops prettytable tabulate

echo ""
echo ">>> Step 3: Copy repo to writable location"
if [ ! -d "/kaggle/working/TrackingMamba" ]; then
    cp -r /kaggle/input/mamba-code/TrackingMamba /kaggle/working/TrackingMamba
    echo "Repo copied to /kaggle/working/TrackingMamba"
else
    echo "Repo already exists in /kaggle/working/"
fi

echo ""
echo ">>> Step 4: Create output directories"
mkdir -p /kaggle/working/output/test/networks
mkdir -p /kaggle/working/output/test/tracking_results
mkdir -p /kaggle/working/output/test/result_plots
mkdir -p /kaggle/working/output/test/segmentation_results
mkdir -p /kaggle/working/output/tensorboard
mkdir -p /kaggle/working/output/checkpoints

echo ""
echo ">>> Step 5: Verify installation"
python -c "
import torch
from mamba_ssm import Mamba
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'GPU:      {torch.cuda.get_device_name(0)}')
print(f'mamba_ssm OK')
print('All checks passed!')
"

echo ""
echo "========================================="
echo "  Installation complete!"
echo "========================================="