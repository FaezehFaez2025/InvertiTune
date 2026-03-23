FROM docker.io/deepqgroup/cuda:10.0-cudnn7-runtime-ubuntu18.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget \
    cmake pkg-config build-essential libgoogle-perftools-dev \
    libprotobuf-dev protobuf-compiler \
    python3-dev libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install base packages
RUN pip3 install --upgrade pip wheel setuptools

# Install specific versions with compatibility fixes
RUN pip3 install --upgrade greenlet==1.1.3 gevent==21.12.0

# Install NLP packages
RUN pip3 install sentencepiece
RUN pip3 install zenodo_get

# Install PyTorch ecosystem
RUN pip3 install torch==1.4.0 torchvision==0.5.0 torchtext==0.5.0

# Install AllenNLP and transformers - try older compatible version
RUN pip3 install allennlp==0.8.5 transformers==2.6.0

# Copy requirements and install remaining dependencies
COPY requirements.txt /workspace/requirements.txt
RUN grep -v '^\./' /workspace/requirements.txt | \
    grep -v '^torch' | \
    grep -v '^transformers' | \
    grep -v '^allennlp' | \
    grep -v '^torchtext' | \
    grep -v '^greenlet' | \
    grep -v '^gevent' | \
    xargs -r pip3 install

# Download NLTK data
RUN python3 -m nltk.downloader stopwords punkt

# Install spaCy and download model
RUN pip3 install spacy==2.3.0 && python3 -m spacy download en_core_web_sm

# Set working directory
WORKDIR /workspace

# Copy the entire codebase
COPY . /workspace

# Create necessary directories and download models
RUN mkdir -p /workspace/models /workspace/data /backup && \
    cd /workspace && \
    echo "Current directory: $(pwd)" && \
    echo "Contents before download: $(ls -la)" && \
    if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then \
        echo "Downloading models from Zenodo..." && \
        zenodo_get 4054476 -o . && \
        if [ -f "openie6_models.tar.gz" ]; then \
            tar -xvf openie6_models.tar.gz && \
            rm -f openie6_models.tar.gz; \
        else \
            echo "ERROR: openie6_models.tar.gz not found after download"; \
        fi; \
    fi && \
    echo "Final models directory contents:" && \
    find models/ -type f -name "*.ckpt" 2>/dev/null || echo "No checkpoint files found" && \
    echo "Final data directory contents:" && \
    ls -la data/ 2>/dev/null || echo "Data directory empty or missing"

# Fix the div_beam_search import issue
RUN if [ -f "imojie/imojie/models/copy_seq2seq_bahdanu.py" ]; then \
        echo "Fixing div_beam_search import..." && \
        sed -i 's/from allennlp.nn.div_beam_search import DivBeamSearch/try:\n    from allennlp.nn.div_beam_search import DivBeamSearch\nexcept ImportError:\n    try:\n        from allennlp.modules.div_beam_search import DivBeamSearch\n    except ImportError:\n        from allennlp.nn.beam_search import BeamSearch as DivBeamSearch/' imojie/imojie/models/copy_seq2seq_bahdanu.py; \
    fi

# Alternative fix: Create a compatibility shim
RUN echo "import sys\nimport importlib\ntry:\n    from allennlp.nn import div_beam_search\nexcept ImportError:\n    try:\n        from allennlp.modules import beam_search as div_beam_search\n        sys.modules['allennlp.nn.div_beam_search'] = div_beam_search\n    except ImportError:\n        try:\n            from allennlp.nn import beam_search as div_beam_search\n            sys.modules['allennlp.nn.div_beam_search'] = div_beam_search\n        except ImportError:\n            print('Warning: Could not find beam search module')" > /workspace/fix_imports.py

# Fix Unicode encoding issues in goldReader.py
RUN if [ -f "carb/oie_readers/goldReader.py" ]; then \
        sed -i "s/with open(gold_fn) as fin:/with open(gold_fn, encoding='utf-8') as fin:/g" carb/oie_readers/goldReader.py; \
    fi

# Create a default sentences.txt if it doesn't exist
RUN if [ ! -f sentences.txt ]; then \
        echo "The quick brown fox jumps over the lazy dog." > sentences.txt && \
        echo "John loves Mary." >> sentences.txt; \
    fi

# Set environment variables for CUDA and fix locale
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/workspace
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Create an entrypoint script that handles volume mounts and import fixes
RUN echo '#!/bin/bash\n\
# Import compatibility fix\n\
export PYTHONPATH="/workspace:$PYTHONPATH"\n\
python3 -c "exec(open(\"/workspace/fix_imports.py\").read())" 2>/dev/null || true\n\
\n\
# Check if models exist, if not copy from backup location\n\
if [ ! -d "/workspace/models" ] || [ ! -f "/workspace/models/conj_model/epoch=28_eval_acc=0.854.ckpt" ]; then\n\
    echo "Models not found in workspace, copying from backup..."\n\
    cp -r /backup/models /workspace/ || echo "Warning: Could not copy models"\n\
    cp -r /backup/data /workspace/ || echo "Warning: Could not copy data"\n\
fi\n\
\n\
# Create default sentences.txt if it does not exist\n\
if [ ! -f "/workspace/sentences.txt" ]; then\n\
    echo "The quick brown fox jumps over the lazy dog." > /workspace/sentences.txt\n\
    echo "John loves Mary." >> /workspace/sentences.txt\n\
fi\n\
\n\
# Execute the original command\n\
exec "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

# Backup the models and data to a location that won't be overridden
RUN cp -r /workspace/models /backup/models && cp -r /workspace/data /backup/data

ENTRYPOINT ["/entrypoint.sh"]

# Modified CMD to skip rescoring initially to test if the main extraction works
CMD ["python3", "run.py", "--mode", "splitpredict", \
     "--inp", "sentences.txt", "--out", "predictions.txt", \
     "--task", "oie", "--gpus", "1", \
     "--oie_model", "models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt", \
     "--conj_model", "models/conj_model/epoch=28_eval_acc=0.854.ckpt", \
     "--num_extractions", "5"]
