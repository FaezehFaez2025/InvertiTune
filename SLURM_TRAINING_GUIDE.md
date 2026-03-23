# SLURM Training Guide: Text-to-KG SFT with Enroot

This guide provides step-by-step instructions for submitting and running the Text-to-KG supervised fine-tuning (SFT) training job on a SLURM cluster using Enroot containers.

---

## Table of Contents

- [Step 1: Create Personal Area for Logs and Experiments](#step-1-create-personal-area-for-logs-and-experiments)
- [Step 2: Copy Your Data/Code (Optional)](#step-2-copy-your-datacode-optional)
- [Step 3: Set Proper Permissions](#step-3-set-proper-permissions)
- [Step 4: Build Docker Image (If Not Already Built)](#step-4-build-docker-image-if-not-already-built)
- [Step 5: Convert Docker Image to Enroot Format](#step-5-convert-docker-image-to-enroot-format)
- [Step 6: List and Verify Enroot Images](#step-6-list-and-verify-enroot-images)
- [Step 7: Create the SLURM Batch Script](#step-7-create-the-slurm-batch-script)
  - [Customizing for Different Tasks](#customizing-for-different-tasks)
- [Step 8: Submit the Job](#step-8-submit-the-job)
- [Step 9: Monitor Your Job](#step-9-monitor-your-job)
- [Step 10: Cancel a Job (If Needed)](#step-10-cancel-a-job-if-needed)
- [Useful SLURM Commands](#useful-slurm-commands)

---

## Step 1: Create Personal Area for Logs and Experiments

First, create directories for storing SLURM logs and your project files:

```bash
mkdir -p /work_shared/$USER/{slurm-logs,projects}
```

---

## Step 2: Copy Your Data/Code (Optional)

If your code is not already in the shared workspace, copy it using rsync:

```bash
rsync -avP ./NeoGraphRAG/ /work_shared/$USER/projects/NeoGraphRAG/
```

---

## Step 3: Set Proper Permissions

Set appropriate permissions for your personal folders (700 or 750 recommended):

```bash
chmod -R 750 /work_shared/$USER
```

---

## Step 4: Build Docker Image (If Not Already Built)

If you haven't built the Docker image yet, build it first. Navigate to the `Text_to_KG_SFT/` directory:

```bash
cd /work_shared/$USER/projects/NeoGraphRAG/NeoGraphRAG/Text_to_KG_SFT
```

Build the Docker image:

```bash
sudo docker build --pull --no-cache -t text2kg-sft:cu121 .
```

**What this does:**
- `--pull`: Pulls the latest base image
- `--no-cache`: Builds from scratch without using cached layers
- `-t text2kg-sft:cu121`: Tags the image with name `text2kg-sft` and tag `cu121`
- `.`: Uses the Dockerfile in the current directory

**Note:** This step takes several minutes.

If you already have the image built, you can skip this step and verify it exists in Step 5.1.

---

## Step 5: Convert Docker Image to Enroot Format

### 5.1 Find Your Docker Image

List available Docker images to find the one you want to convert (**Note:** `sudo` is required):

```bash
sudo docker images
```

**Example output:**

```
REPOSITORY       TAG                                                            IMAGE ID       CREATED         SIZE
text2kg-sft      cu121                                                          b64f683d0646   8 hours ago     21.7GB
trl-env          cu121                                                          6b7c041b3f19   31 hours ago    6.87GB
<none>           <none>                                                         b2cef647c3b3   2 days ago      6.87GB
llmpretrain      latest                                                         f1f20517ef61   6 weeks ago     18GB
ubuntu           22.04                                                          392fa14dddd0   6 weeks ago     77.9MB
```

In this example, we'll use the **`text2kg-sft:cu121`** image for our training (built in Step 4).

### 5.2 Check Module Availability

Verify that the docker2enroot conversion tool is available:

```bash
module avail
```

Look for `enroot/docker2enroot` in the output.

### 5.3 Load the Conversion Tool

Load the enroot module with docker2enroot:

```bash
module load enroot/docker2enroot
```

### 5.4 Convert Docker Image to Enroot Image

Convert the `text2kg-sft:cu121` Docker image to Enroot format:

```bash
sudo -E $(which docker2enroot) --image text2kg-sft --tag cu121
```

**Note:** The conversion process will create an enroot image named `text2kg-sft-cu121` that can be used across SLURM jobs.

---

## Step 6: List and Verify Enroot Images

Verify that your image was successfully converted:

```bash
enroot list
```

**Example output:**

```
text2kg-sft-cu121
trl-env-cu121
```

You should see your converted image `text2kg-sft-cu121` in the list.

---

## Step 7: Create the SLURM Batch Script

Create a new `.sbatch` file for submitting your training job:

```bash
nano run_sft_training_enroot.sbatch
```

**Copy and paste the following content into the file:**

**Note:** This script is configured to use the `text2kg-sft-cu121` enroot image we converted in Step 5.

```bash
#!/bin/bash -l
#SBATCH --job-name=sft-train
#SBATCH --partition=compute
#SBATCH --account=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work_shared/%u/slurm-logs/%x-%j.out
#SBATCH --error=/work_shared/%u/slurm-logs/%x-%j.err

# --- robust shell ---
set -euo pipefail

# --- define a safe node-local temp if SLURM_TMPDIR is not provided ---
export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/slurm-${SLURM_JOB_ID:-$$}}"
mkdir -p "$SLURM_TMPDIR"

# --- paths & image names ---
PROJ_ROOT="/work_shared/$USER/projects/NeoGraphRAG/NeoGraphRAG"
ENROOT_LIBRARY="/work_shared/enroot_lib"
IMAGE_NAME="text2kg-sft-cu121"
IMAGE_SRC_DIR="$ENROOT_LIBRARY/$IMAGE_NAME"
IMAGE_RUN_DIR="$SLURM_TMPDIR/enroot/data/$IMAGE_NAME"      # per-node copy (avoids permission/lock issues)

# --- caches (persist models/downloads in your project) ---
export HF_HOME="$PROJ_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

echo "== JOB START $(date) =="
echo "Node: $(hostname)"
nvidia-smi -L || true

# --- make node-local enroot dirs and copy the image in ---
export ENROOT_RUNTIME_PATH="$SLURM_TMPDIR/enroot/run"
export ENROOT_DATA_PATH="$SLURM_TMPDIR/enroot/data"
export ENROOT_CACHE_PATH="$SLURM_TMPDIR/enroot/cache"
mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"

echo "Copying enroot image to node-local tmp (if needed)…"
if [ ! -d "$IMAGE_SRC_DIR" ]; then
  echo "ERROR: Enroot image not found: $IMAGE_SRC_DIR" >&2
  exit 1
fi
mkdir -p "$(dirname "$IMAGE_RUN_DIR")"
rsync -a \
  --exclude='var/cache/apt/' \
  --exclude='var/cache/ldconfig' \
  --exclude='var/cache/debconf/passwords.dat' \
  --exclude='var/lib/dpkg/lock*' \
  --exclude='var/lib/dpkg/triggers/Lock' \
  --exclude='var/log/btmp' \
  --exclude='var/log/apt/' \
  --exclude='etc/ssl/private/' \
  --exclude='etc/.pwd.lock' \
  --exclude='etc/gshadow' \
  --exclude='etc/gshadow-' \
  --exclude='etc/shadow' \
  --exclude='etc/security/opasswd' \
  --exclude='root/' \
  "$IMAGE_SRC_DIR/" "$IMAGE_RUN_DIR/" || echo "rsync finished with code $? (some files skipped)"

echo "ENROOT paths:"
echo "  ENROOT_RUNTIME_PATH=$ENROOT_RUNTIME_PATH"
echo "  ENROOT_DATA_PATH=$ENROOT_DATA_PATH"
echo "  ENROOT_CACHE_PATH=$ENROOT_CACHE_PATH"
echo "  Using image: $(basename "$IMAGE_RUN_DIR")"

# --- launch container with enroot; use python -m llamafactory.cli instead of llamafactory-cli ---
srun --exclusive \
enroot start --rw \
  --root \
  --mount "$PROJ_ROOT:/workspace" \
  --mount "/work_shared:/work_shared" \
  --env "HF_HOME=$HF_HOME" \
  --env "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE" \
  "$IMAGE_NAME" \
bash -lc '
set -euo pipefail

echo "Inside container on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

cd /workspace/Text_to_KG_SFT/LLaMA-Factory

# === Step 1: Remove saves folder ===
echo "Removing saves folder in LLaMA-Factory..."
rm -rf saves

# === Step 2: Run training ===
echo "Running training..."
export FORCE_TORCHRUN=1
python -m llamafactory.cli train examples/train_full/qwen2.5_full_sft.yaml
'

echo "== JOB END $(date) =="
```

**Important:** The script is already configured for the `text2kg-sft:cu121` image. However, verify/adjust these variables if your setup differs:

- `IMAGE_NAME="text2kg-sft-cu121"` - **Already set correctly** for the image from Step 5
- `PROJ_ROOT="/work_shared/$USER/projects/NeoGraphRAG/NeoGraphRAG"` - Adjust if your project is in a different location
- `ENROOT_LIBRARY="/work_shared/enroot_lib"` - Location where enroot images are stored (check with your cluster admin)
- `#SBATCH --partition=compute` - Change to your cluster's partition name if different
- `#SBATCH --account=main` - Change to your SLURM account name if different

 ---

### Customizing for Different Tasks

The following parts can be customized based on your specific needs:

**1. Job Name**

```bash
#SBATCH --job-name=sft-train
```

Replace `sft-train` with any name you want. This name will appear when you check job status with `squeue` and in your log filenames.

**2. GPU Resources**

```bash
#SBATCH --gres=gpu:1
```

Specifies the number of GPUs to allocate. Change `1` to the number of GPUs you need (e.g., `gpu:2`, `gpu:4`, `gpu:8`). 

To check total GPUs per node: `scontrol show node | grep -E "NodeName|Gres"`

To check GPU usage by running jobs: `squeue -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R %b"`

**3. CPU Cores**

```bash
#SBATCH --cpus-per-task=8
```

Specifies the number of CPU cores to allocate.

**4. System Memory (RAM)**

```bash
#SBATCH --mem=64G
```

Specifies the amount of system memory (RAM) to allocate, not GPU memory. GPU memory is fixed by hardware (e.g., H100 has 80GB).

**5. Time Limit**

```bash
#SBATCH --time=02:00:00
```

Sets the maximum wall-clock time your job can run (format: `HH:MM:SS`). If your job exceeds this time, it will be automatically terminated. This helps the scheduler plan resources and prevents runaway jobs from blocking the queue. Jobs with shorter time limits may be scheduled faster.

**6. Project Root Path**

```bash
PROJ_ROOT="/work_shared/$USER/projects/NeoGraphRAG/NeoGraphRAG"
```

Specifies the path to your project directory on the shared filesystem. Change this to match where your code is located. This path will be mounted inside the container as `/workspace`.

**7. Enroot Image Name**

```bash
IMAGE_NAME="text2kg-sft-cu121"
```

Specifies the name of the enroot image to use (created in Step 5). Change this to match your converted enroot image name. Check available images with `enroot list`.

**8. Execution Code Inside Container**

Starting from `cd /workspace/...` to the end of the script, replace with whatever commands you want to execute.

---

## Step 8: Submit the Job

Submit your training job to the SLURM scheduler:

```bash
sbatch run_sft_training_enroot.sbatch
```

You should see output like:

> Submitted batch job 1234

The number (e.g., 1234) is your job ID.

---

## Step 9: Monitor Your Job

### Check Job Status

View your running/queued jobs:

```bash
squeue -u $USER
```

Output example:
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
 1352   compute sft-trai   faezeh  R      28:41      1 sines-2-embody-ai-node-2
```

Status codes:
- `PD` = Pending (waiting for resources)
- `R` = Running

### Monitor Output Log (Real-time)

Watch the standard output log in real-time:

```bash
tail -f /work_shared/$USER/slurm-logs/sft-train-<JOB_ID>.out
```

Replace `<JOB_ID>` with your actual job ID. Press `Ctrl+C` to stop following.

### Monitor Error Log (Real-time)

Watch the error log in real-time:

```bash
tail -f /work_shared/$USER/slurm-logs/sft-train-<JOB_ID>.err
```

---

## Step 10: Cancel a Job (If Needed)

If you need to cancel a running or pending job:

```bash
scancel <JOB_ID>
```

Replace `<JOB_ID>` with your actual job ID.

---

## Useful SLURM Commands

- Check job details: `scontrol show job <JOB_ID>`
- List all your jobs: `sacct -u $USER`
- Check cluster info: `sinfo`