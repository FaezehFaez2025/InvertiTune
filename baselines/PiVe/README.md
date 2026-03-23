# PiVe
This is the official code for the ACL 2024 paper: [PiVe: Prompting with Iterative Veriﬁcation Improving Graph-based Generative Capability of LLMs](https://aclanthology.org/2024.findings-acl.400.pdf).

## Files Introduction
1. `GenWiki-HIQ` is the created dataset using verifier module, which contains 110K parallel graph-text pairs.
2. `data_processing_script` contains `data_process.ipynb` to create the training data for the verifier module and test data for each iteration.
3. `datasets` contains the used kelm-sub and webnlg+2020 datasets. `pive_verifier_training_data.zip` contains the generated verifier training data for single verifier module and unified verifier module, which can be directly used to train the verifier modules.
4. `graph_evaluation` contains the graph evaluation metrics.
5. `prompt_scripts`contains the sctipts to prompt LLMs.
6. `single_verifier` contains the training sctipt for single verifier using T5-Large.
7. `unified_verifier` contains the training sctipt for unified verifier using insturction-tuning on Flan-T5-XXL.

## Clarification and Guidance 
For the file "data/only_one_error_webnlg/train.source" which is the training data for the verifier module, you need to use the first section of our provided data_process.ipynb to manually generate. We also upload the generated verifier training data in `pive_verifier_training_data.zip` for your convenience.

For the file "GPT3.5_result_KELM/test.target" in `run_chatgpt.py`, it is the same as the file which path is `datasets/kelm_sub/test.target`. You can just copy it to a folder like `GPT3.5_result_KELM` or use your own folder name, and put the corresponding file path in `run_chatgpt.py`. Then you can run the `run_chatgpt.py` to prompt LLMs for graph generation. After getting the results from LLMs, you need to use our `data_process.ipynb` to create the input for the single/unified verifier module from the generated graph. Then you can feed the input to the trained verifier module to predict the missing triple. For subsequent iterations, remember to set `iteration1 = False` in the `run_chatgpt.py` when prompting the LLMs.

## Citation
```
@inproceedings{han-etal-2024-pive,
    title = "{P}i{V}e: Prompting with Iterative Verification Improving Graph-based Generative Capability of {LLM}s",
    author = "Han, Jiuzhou  and
      Collier, Nigel  and
      Buntine, Wray  and
      Shareghi, Ehsan",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.400",
    pages = "6702--6718",
    abstract = "Large language models (LLMs) have shown great abilities of solving various natural language tasks in different domains. Due to the training objective of LLMs and their pre-training data, LLMs are not very well equipped for tasks involving structured data generation. We propose a framework, Prompting with Iterative Verification (PiVe), to improve graph-based generative capability of LLMs. We show how a small language model could be trained to act as a verifier module for the output of an LLM(i.e., ChatGPT, GPT-4), and to iteratively improve its performance via fine-grained corrective instructions. We also show how the verifier module could apply iterative corrections offline for a more cost-effective solution to the text-to-graph generation task. Experiments on three graph-based datasets show consistent improvement gained via PiVe. Additionally, we create GenWiki-HIQ and highlight that the verifier module can be used as a data augmentation tool to help improve the quality of automatically generated parallel text-graph datasets.",
}
```
# Additional Instructions for Running the Pipeline
## Setup and Usage Guide

### **Recommended Hardware Specifications**
- 1 × NVIDIA A100 PCIe 80GB
- 8 vCPUs
- 32 GB memory

### **Step 1: Setup Python Environment and CUDA**

```bash
conda create -n pive python=3.8 -y && \
conda activate pive && \
conda install cudatoolkit=11.1 -c conda-forge -y && \
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html && \
pip install transformers==3.3.1 pytorch-lightning==0.9.0 numpy==1.19.5 protobuf==3.20.0 parsimonious==0.10.0 gitpython sacrebleu unidecode networkx rouge-score openai==0.28 notebook ipykernel pandas==1.4.0 python-dotenv
```

```bash
nvidia-smi -mig 0
```

```bash
reboot
```

### **Step 2: Download Pre-trained Models**

```bash
# Create models directory
mkdir -p models/t5-large
cd models/t5-large

# Download T5-Large model files using wget (more reliable for large files)
wget -c https://huggingface.co/t5-large/resolve/main/config.json
wget -c https://huggingface.co/t5-large/resolve/main/pytorch_model.bin
wget -c https://huggingface.co/t5-large/resolve/main/spiece.model
wget -c https://huggingface.co/t5-large/resolve/main/tokenizer.json
```

### **Step 3: Setup Training Data**

```bash
# First, extract the training data from the zip file
cd ../../datasets
unzip pive_verifier_training_data.zip

# Navigate to single_verifier directory
cd ../single_verifier

# Create data directory and copy training data
mkdir -p data
cp -r ../datasets/pive_verifier_training_data/single_verifier_training_data/* data/
```

### **Step 4: Train Single Verifier**

```bash
# Start training with T5-Large
bash run_finetune_kelm.sh ../models/t5-large 0
```

### **Step 5: Setup Environment for Graph Generation**

0. Navigate back to the main PiVe directory:
```bash
cd ..
```

1. Set up the initial directory structure and copy test data (run from main PiVe directory):
```bash
# Copy test data from KELM dataset
mkdir -p prompt_scripts/GPT3.5_result_KELM && cp datasets/kelm_sub/test.target prompt_scripts/GPT3.5_result_KELM/
```

To use custom data, put your JSON file in `../LLaMA-Factory/data/T2G_test.json` and run:
```bash
cd prompt_scripts && python extract_inputs.py && cd ..
```

2. The OpenAI API key should be configured in `prompt_scripts/.env` file:
```bash
nano prompt_scripts/.env  # Add your OpenAI API key here
```

3. Set the checkpoint for verifier step:
```bash
# Set checkpoint filename for verifier
CKPT_NAME="val_avg_loss=0.0725-step_count=4.ckpt"
```

### **Step 6: Generate and Verify Graphs**

Set the iteration number:
```bash
x=1
```

1. Generate graphs using ChatGPT:
```bash
# Navigate to prompt_scripts directory
cd prompt_scripts

# Run iteration
python run_chatgpt.py --iteration $x
```

Optional: Specify a custom output directory (default: GPT3.5_result_KELMs):
```bash
python run_chatgpt.py --iteration $x --output_dir custom_output_dir
```

2. Process the generated graphs:
This step processes the ChatGPT-generated graphs and prepares them for the verifier by combining the generated graphs with test texts to create the required `test.source` file.
```bash
python process_data.py --iteration $x
```

3. Create necessary dummy files for the verifier:
This step creates placeholder files (train.source, train.target, val.source, val.target) required by the verifier's code structure. These files do not affect the verification results since we are only using the prediction mode.
```bash
python create_dummy_files.py --iteration $x
cd ..
```

4. Run the verifier:
```bash
cd single_verifier

# Run verifier prediction
python finetune.py --model_name_or_path outputs/t5_large_only_one_error_kelm_new/best_tfmr \
    --checkpoint outputs/t5_large_only_one_error_kelm_new/${CKPT_NAME} \
    --eval_batch_size 8 \
    --data_dir ../prompt_scripts/GPT3.5_result_KELMs/Iteration$x \
    --output_dir inference_results \
    --eval_max_gen_length 128 \
    --do_predict \
    --eval_beams 5 \
    --task graph2text \
    --gpus 1
```

5. Copy verifier results:
```bash
# Save results for next iteration
cp inference_results/test_generations.txt ../prompt_scripts/GPT3.5_result_KELMs/Iteration$x/verifier_result.txt
cd ..
```

### **Step 7: Subsequent Iterations**

For subsequent iterations, repeat Step 6 with a different value for x (e.g., x=2 for the second round).

### **Step 8: Move Results**

Set the iteration number:
```bash
x=1
```

To move the results from a specific iteration into a structured directory:
```bash
cd prompt_scripts

# Move results from specified iteration
python copy_results.py --iteration $x --base_dir GPT3.5_result_KELMs
```

This will create a `results/original_prediction` directory in the prompt_scripts directory and copy the `test_generated_graphs.txt` file as `aggregated_pive_triplets.txt`.

### **Step 9: Triples Post-processing**

You have two options at this stage:

**(a) Find common triples across methods (default)**  
To find common triples across all files in the `results` directory:
```bash
python find_common_triples.py
```
This will create a `results/common_triples` directory and save files containing only the predictions that are non-empty by all methods in the `results/original_prediction` directory.

**(b) Clean malformed triples for separate analysis**  
If you only want to sanitize malformed or unparsable lines in `aggregated_pive_triplets.txt`, you can run:
```bash
python fix_aggregated_pive_triplets.py \
  --input results/original_prediction/aggregated_pive_triplets.txt \
  --output results/original_prediction/aggregated_pive_triplets_sanitized.txt
```
This produces a cleaned copy of the triples file for inspection or further processing.

### **Step 10: Computing Metrics**

First, create and setup the evaluation environment:
```bash
conda create -n pive_eval python=3.8 -y && conda activate pive_eval && pip install bert-score nltk spacy==3.4.4 scikit-learn scipy networkx rouge-score sacrebleu numpy==1.24.3 && python -m spacy download en_core_web_sm
```

Then, post-process the prediction file to fix format issues:
```bash
python post_process_triples.py --input_file "results/common_triples/aggregated_pive_triplets.txt"
```

Then, set the required environment variables:
```bash
export RESULT_DIR="results/common_triples"
```

```bash
export PRED_FILE="aggregated_pive_triplets_post_processed.txt"
```

```bash
export GOLD_FILE="aggregated_ground_truth_triplets.txt"
```

Then compute the evaluation metrics:
```bash
python ../graph_evaluation/metrics/eval.py \
    --pred_file "$RESULT_DIR/$PRED_FILE" \
    --gold_file "$RESULT_DIR/$GOLD_FILE"
```

## Preparing Controlled Extraction (CE) Data for PiVe

If you want to get results from the controlled extraction dataset, you can use the `prepare_ce_data_for_pive.py` script to process the test folders and prepare the data for PiVe pipeline.

### **Usage:**

```bash
cd prompt_scripts

# Process folders 1 through N (replace N with desired number)
python prepare_ce_data_for_pive.py N
```

### **What it does:**
- Processes test folders from `result/controlled_extraction/test/1/` to `result/controlled_extraction/test/N/`
- Reads `text.txt` from each folder and removes all newlines
- Clears and populates `GPT3.5_result_KELM/test.target` with the processed text
- Each processed text becomes a single line in the target file

After running this script, you can proceed with the PiVe pipeline starting from Step 6 (Generate and Verify Graphs) using the prepared CE data.