# Docker Setup for Text-to-KG SFT Training

## Build Docker Image

From the `Text_to_KG_SFT/` directory:

```bash
docker build --pull --no-cache -t text2kg-sft:cu121 .
```

---

## Run Training

From the `Text_to_KG_SFT/` directory, navigate to repository root:

```bash
cd ../..
```

Set up HuggingFace cache directory:

```bash
export HF_HOME=$PWD/NeoGraphRAG/.cache/huggingface
mkdir -p "$HF_HOME"
```

```bash
docker run --rm --gpus all \
  -e HF_HOME \
  -e PYTHONPATH=/workspace \
  -v "$PWD/NeoGraphRAG":/workspace \
  -w /workspace/Text_to_KG_SFT \
  text2kg-sft:cu121 \
  bash -lc 'rm -rf LLaMA-Factory/saves && cd LLaMA-Factory && CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2.5_full_sft.yaml'
```

---

