# Dockerfile for Text-to-KG SFT Training
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install basic tools
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Install PyTorch and dependencies in clear steps
RUN set -eux; \
    pip install --upgrade pip; \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.1.0+cu121 \
        torchvision==0.16.0+cu121 \
        torchaudio==2.1.0+cu121; \
    pip install \
        deepspeed==0.15.0 \
        accelerate==0.34.0 \
        transformers==4.49.0 \
        tokenizers==0.21.0 \
        llamafactory==0.9.3 \
        bert-score==0.3.13 \
        peft==0.15.0 \
        trl==0.9.6 \
        datasets==3.5.0 \
        huggingface-hub==0.33.2 \
        tensorboard \
        sentence-transformers \
        python-dotenv \
        openai

WORKDIR /workspace
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

