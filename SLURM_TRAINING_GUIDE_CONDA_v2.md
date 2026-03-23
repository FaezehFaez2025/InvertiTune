# SLURM Training Guide: Text-to-KG SFT with Conda

## Table of Contents

- [Step 1: Create Personal Area for Logs and Experiments](#step-1-create-personal-area-for-logs-and-experiments)
- [Step 2: Copy Your Project to the Shared Location](#step-2-copy-your-project-to-the-shared-location)
- [Step 3: Set Up Conda Environment Directory](#step-3-set-up-conda-environment-directory)
- [Step 4: Set Up Conda Environment](#step-4-set-up-conda-environment)
  - [4.1: Initialize Conda](#41-initialize-conda)
  - [4.2: Create Conda Environment](#42-create-conda-environment)
  - [4.3: Activate Environment and Install Dependencies](#43-activate-environment-and-install-dependencies)
  - [4.4: Verify Installation](#44-verify-installation)
- [Step 5: Create the SLURM Batch Script](#step-5-create-the-slurm-batch-script)
- [Step 6: Submit the Job](#step-6-submit-the-job)
- [Step 7: Monitor Your Job](#step-7-monitor-your-job)
- [Step 8: Cancel a Job (If Needed)](#step-8-cancel-a-job-if-needed)
- [Useful SLURM Commands](#useful-slurm-commands)
- [Troubleshooting](#troubleshooting)

---

## Step 1: Create Personal Area for Logs and Experiments

```bash
mkdir -p /pm/$USER/slurm-logs
mkdir -p /pm/$USER/projects
```

---

## Step 2: Copy Your Project to the Shared Location

```bash
rsync -avP ./NeoGraphRAG/ /pm/$USER/projects/NeoGraphRAG/
```

---

## Step 3: Set Up Conda Environment Directory

```bash
mkdir -p /pm/$USER/conda_envs
```

---

## Step 4: Set Up Conda Environment

### 4.1: Initialize Conda

```bash
source /pm/miniconda3/etc/profile.d/conda.sh
```

### 4.2: Create Conda Environment

```bash
conda create -p /pm/$USER/conda_envs/text2kg-sft python=3.10 -y
```

### 4.3: Activate Environment and Install Dependencies

**Activate the conda environment:**

```bash
conda activate /pm/$USER/conda_envs/text2kg-sft
export PYTHONNOUSERSITE=1
```

**Install PyTorch with CUDA support:**

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**Install remaining packages from PyPI:**

```bash
pip install deepspeed==0.15.0 accelerate==0.34.0 transformers==4.49.0 tokenizers==0.21.0 llamafactory==0.9.3 bert-score==0.3.13 peft==0.15.0 trl==0.9.6 datasets==3.5.0 huggingface-hub==0.33.2 tensorboard
```

### 4.4: Verify Installation

Verify that LLaMA-Factory is installed correctly:

```bash
python -c "import llamafactory; print('LLaMA-Factory installed successfully')"
```

Verify PyTorch and CUDA:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Step 5: Create the SLURM Batch Script

Create a new `.sbatch` file for submitting your training job:

```bash
nano run_sft_training_conda.sbatch
```

Copy and paste the following content into the file:

```bash
#!/bin/bash -l
#SBATCH --job-name=sft-train
#SBATCH --partition=compute
#SBATCH --account=main
#SBATCH --qos=mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pm/%u/slurm-logs/%x-%j.out
#SBATCH --error=/pm/%u/slurm-logs/%x-%j.err

# --- robust shell ---
set -euo pipefail

# --- paths ---
PROJ_ROOT="/pm/$USER/projects/NeoGraphRAG/NeoGraphRAG"
CONDA_ENV_NAME="text2kg-sft"

# --- caches (persist models/downloads in your project) ---
export HF_HOME="$PROJ_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$PROJ_ROOT/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

echo "== JOB START $(date) =="
echo "Node: $(hostname)"
echo "User: $USER"
nvidia-smi -L || true

# --- initialize conda ---
# Try module system first, then fall back to direct path
if command -v module >/dev/null 2>&1; then
    module load conda 2>/dev/null || true
fi

# Initialize conda from known locations (prioritize /pm since $HOME may not exist on compute nodes)
if [ -f "/pm/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/pm/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/usr/local/anaconda3/etc/profile.d/conda.sh"
elif [ -n "${HOME:-}" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -n "${HOME:-}" ] && [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    # Try to find conda in PATH (if module loaded it)
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            echo "ERROR: Could not find conda. Please adjust the conda initialization path in the script." >&2
            exit 1
        fi
    else
        echo "ERROR: Could not find conda. Please adjust the conda initialization path in the script." >&2
        exit 1
    fi
fi

# --- activate conda environment ---
echo "Available conda environments:"
conda env list || echo "Could not list environments"

# Check if environment exists in user's own space first
USER_ENV_DIR="/pm/$USER/conda_envs/$CONDA_ENV_NAME"
if [ -d "$USER_ENV_DIR" ] && [ -f "$USER_ENV_DIR/bin/python" ]; then
    echo "Found environment in user space: $USER_ENV_DIR"
    # Activate using full path
    source "$USER_ENV_DIR/bin/activate" || {
        export PATH="$USER_ENV_DIR/bin:$PATH"
        echo "Activated environment using PATH method"
    }
# Check if environment exists in shared users location (if user is in conda_users group)
elif [ -d "/pm/miniconda3/envs/users/$USER/$CONDA_ENV_NAME" ] && [ -f "/pm/miniconda3/envs/users/$USER/$CONDA_ENV_NAME/bin/python" ]; then
    USER_ENV_DIR="/pm/miniconda3/envs/users/$USER/$CONDA_ENV_NAME"
    echo "Found environment in shared users location: $USER_ENV_DIR"
    source "$USER_ENV_DIR/bin/activate" || {
        export PATH="$USER_ENV_DIR/bin:$PATH"
        echo "Activated environment using PATH method"
    }
else
    echo "Attempting to activate environment: $CONDA_ENV_NAME"
    conda activate "$CONDA_ENV_NAME" || {
        echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'" >&2
        echo "" >&2
        echo "The environment does not exist. You need to create it first." >&2
        echo "" >&2
        echo "To create the environment in your own space, run on the login node:" >&2
        echo "  source /pm/miniconda3/etc/profile.d/conda.sh" >&2
        echo "  mkdir -p /pm/$USER/conda_envs" >&2
        echo "  conda create -p /pm/$USER/conda_envs/$CONDA_ENV_NAME python=3.10 -y" >&2
        echo "  conda activate /pm/$USER/conda_envs/$CONDA_ENV_NAME" >&2
        echo "  cd $PROJ_ROOT/Text_to_KG_SFT/LLaMA-Factory" >&2
        echo "  pip install -e \".[torch,metrics]\"" >&2
        echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" >&2
        echo "" >&2
        exit 1
    }
fi

# --- verify environment ---
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

# --- change to project directory ---
cd "$PROJ_ROOT/Text_to_KG_SFT/LLaMA-Factory"

# --- Step 1: Remove saves folder (optional, for clean start) ---
echo "Removing saves folder in LLaMA-Factory (if exists)..."
rm -rf saves || true

# --- Step 2: Run training ---
echo "Running training..."
export FORCE_TORCHRUN=1
python -m llamafactory.cli train examples/train_full/qwen2.5_full_sft.yaml

echo "== JOB END $(date) =="
```

---

## Step 6: Submit the Job

```bash
sbatch run_sft_training_conda.sbatch
```

---

## Step 7: Monitor Your Job

### Check Job Status

```bash
squeue -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R %b"
```

### Monitor Output Log

```bash
tail -f /pm/$USER/slurm-logs/sft-train-<JOB_ID>.out
```

### Monitor Error Log

```bash
tail -f /pm/$USER/slurm-logs/sft-train-<JOB_ID>.err
```

---

## Step 8: Cancel a Job

```bash
scancel <JOB_ID>
```