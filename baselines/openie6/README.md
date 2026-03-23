# OpenIE6 System 

This repository contains the code for the paper:\
OpenIE6: Iterative Grid Labeling and Coordination Analysis for Open Information Extraction\
Keshav Kolluru*, Vaibhav Adlakha*, Samarth Aggarwal, Mausam and Soumen Chakrabarti\
EMNLP 2020

\* denotes equal contribution

## Installation
```
conda create -n openie6 python=3.6
conda activate openie6
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m nltk.downloader punkt 
```

All results have been obtained on V100 GPU with CUDA 10.0
NOTE: HuggingFace transformers==2.6.0 is necessary. The latest version has a breaking change in the way tokenizer is used in the code. It will not raise an error but will give wrong results!

## Download Resources
Download Data (50 MB)
```
zenodo_get 4054476
tar -xvf openie6_data.tar.gz
```

Download Models (6.6 GB)
```
zenodo_get 4055395
tar -xvf openie6_models.tar.gz
```
<!-- wget www.cse.iitd.ac.in/~kskeshav/oie6_models.tar.gz
tar -xvf oie6_models.tar.gz
wget www.cse.iitd.ac.in/~kskeshav/oie6_data.tar.gz
tar -xvf oie6_data.tar.gz
wget www.cse.iitd.ac.in/~kskeshav/rescore_model.tar.gz
tar -xvf rescore_model.tar.gz
mv rescore_model models/ -->

## Running Model

New command:
```
python run.py --mode splitpredict --inp sentences.txt --out predictions.txt --rescoring --task oie --gpus 1 --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 
```

Expected models: \
models/conj_model: Performs coordination analysis \
models/oie_model: Performs OpenIE extraction \
models/rescore_model: Performs the final rescoring 

--inp sentences.txt - File with one sentence in each line 
--out predictions.txt - File containing the generated extractions

gpus - 0 for no GPU, 1 for single GPU

Additional flags -
```
--type labels // outputs word-level aligned labels to the file path `out`+'.labels'
--type sentences // outputs decomposed sentences to the file path `out`+'.sentences'
```

Additional Notes:

1. The model is trained with tokenized sentences and hence requires tokenized sentences during prediction as well. The code currently uses nltk tokenization for this purpose. This will lead to the final sentences being different from the input sentences, as they will be the tokenized version. If this is not desirable you can comment the nltk tokenization in data.py and make sure that your sentences are tokenized beforehand.
2. Due to an artifact of training data in conjunction model, it requires the sentence to end with full stop to function correctly.   


## Training Model

### Warmup Model
Training:
```
python run.py --save models/warmup_oie_model --mode train_test --model_str bert-base-cased --task oie --epochs 30 --gpus 1 --batch_size 24 --optimizer adamW --lr 2e-05 --iterative_layers 2
```

Testing:
```
python run.py --save models/warmup_oie_model --mode test --batch_size 24 --model_str bert-base-cased --task oie --gpus 1
```
Carb F1: 52.4, Carb AUC: 33.8


Predicting
```
python run.py --save models/warmup_oie_model --mode predict --model_str bert-base-cased --task oie --gpus 1 --inp sentences.txt --out predictions.txt
```

Time (Approx): 142 extractions/second

### Constrained Model
Training
```
python run.py --save models/oie_model --mode resume --model_str bert-base-cased --task oie --epochs 16 --gpus 1 --batch_size 16 --optimizer adam --lr 5e-06 --iterative_layers 2 --checkpoint models/warmup_oie_model/epoch=13_eval_acc=0.544.ckpt --constraints posm_hvc_hvr_hve --save_k 3 --accumulate_grad_batches 2 --gradient_clip_val 1 --multi_opt --lr 2e-5 --wreg 1 --cweights 3_3_3_3 --val_check_interval 0.1
```

Testing
```
python run.py --save models/oie_model --mode test --batch_size 16 --model_str bert-base-cased --task oie --gpus 1 
```
Carb F1: 54.0, Carb AUC: 35.7

Predicting
```
python run.py --save models/oie_model --mode predict --model_str bert-base-cased --task oie --gpus 1 --inp sentences.txt --out predictions.txt
```

Time (Approx): 142 extractions/second

NOTE: Due to a bug in the code, [link](https://github.com/dair-iitd/openie6/issues/10), we end up using a loss function based only on the constrained loss term and not the original Cross Entropy (CE) loss. It still seems to work well as the warmup model is already trained with the CE loss and the constrained training is initialized from the warmup model.

### Running Coordination Analysis
```
python run.py --save models/conj_model --mode train_test --model_str bert-large-cased --task conj --epochs 40 --gpus 1 --batch_size 32 --optimizer adamW --lr 2e-05 --iterative_layers 2
```

F1: 87.8

### Final Model

Running
```
python run.py --mode splitpredict --inp carb/data/carb_sentences.txt --out models/results/final --rescoring --task oie --gpus 1 --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 
python utils/oie_to_allennlp.py --inp models/results/final --out models/results/final.carb
python carb/carb.py --allennlp models/results/final.carb --gold carb/data/gold/test.tsv --out /dev/null
```
Carb F1: 52.7, Carb AUC: 33.7
Time (Approx): 31 extractions/second

Evaluate using other metrics (Carb(s,s), Wire57 and OIE-16)
```
bash carb/evaluate_all.sh models/results/final.carb carb/data/gold/test.tsv
```

Carb(s,s): F1: 46.4, AUC: 26.8
Carb(s,m) ==> Carb: F1: 52.7, AUC: 33.7
OIE16: F1: 65.6, AUC: 48.4
Wire57: F1: 40.0

## CITE
If you use this code in your research, please cite:

```
@inproceedings{kolluru&al20,
    title = "{O}pen{IE}6: {I}terative {G}rid {L}abeling and {C}oordination {A}nalysis for {O}pen {I}nformation {E}xtraction",\
    author = "Kolluru, Keshav  and
      Adlakha, Vaibhav and
      Aggarwal, Samarth and
      Mausam, and
      Chakrabarti, Soumen",
    booktitle = "The 58th Annual Meeting of the Association for Computational Linguistics (ACL)",
    month = July,
    year = "2020",
    address = {Seattle, U.S.A}
}
```


## LICENSE

Note that the license is the full GPL, which allows many free uses, but not its use in proprietary software which is distributed to others. For distributors of proprietary software, you can contact us for commercial licensing.

## CONTACT

In case of any issues, please send a mail to ```keshav.kolluru (at) gmail (dot) com```

# OpenIE6 Setup Guide

## Prerequisites
- CUDA 10.0 compatible environment

## Step-by-Step Setup

### 1. Download Required Data and Models
**Download the data (50 MB):**
```bash
# Install zenodo_get if not available
pip install zenodo_get

# Download data
zenodo_get 4054476
tar -xvf openie6_data.tar.gz
```

**Download the models (6.6 GB):**
```bash
# Download models
zenodo_get 4055395
tar -xvf openie6_models.tar.gz
```

### 2. Build the Docker Image
```bash
docker build -t openie6:cuda10 .
```

### 2b. Pre-download BERT model weights

The BERT weights must be pre-downloaded on the host before running the container.

```bash
mkdir -p data/pretrained_cache/bert-base-cased data/pretrained_cache/bert-large-cased

wget -c "https://huggingface.co/bert-base-cased/resolve/main/config.json"           -O data/pretrained_cache/bert-base-cased/config.json
wget -c "https://huggingface.co/bert-base-cased/resolve/main/tokenizer_config.json" -O data/pretrained_cache/bert-base-cased/tokenizer_config.json
wget -c "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt"             -O data/pretrained_cache/bert-base-cased/vocab.txt
wget -c "https://huggingface.co/bert-base-cased/resolve/main/special_tokens_map.json" -O data/pretrained_cache/bert-base-cased/special_tokens_map.json
wget -c "https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin"     -O data/pretrained_cache/bert-base-cased/pytorch_model.bin

wget -c "https://huggingface.co/bert-large-cased/resolve/main/config.json"           -O data/pretrained_cache/bert-large-cased/config.json
wget -c "https://huggingface.co/bert-large-cased/resolve/main/tokenizer_config.json" -O data/pretrained_cache/bert-large-cased/tokenizer_config.json
wget -c "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt"             -O data/pretrained_cache/bert-large-cased/vocab.txt
wget -c "https://huggingface.co/bert-large-cased/resolve/main/special_tokens_map.json" -O data/pretrained_cache/bert-large-cased/special_tokens_map.json
wget -c "https://huggingface.co/bert-large-cased/resolve/main/pytorch_model.bin"     -O data/pretrained_cache/bert-large-cased/pytorch_model.bin
```

### 3. Create Test Input Data
```bash
echo "The quick brown fox jumps over the lazy dog." > sentences.txt
echo "John loves Mary and goes to school." >> sentences.txt
echo "Barack Obama was born in Hawaii." >> sentences.txt
```

### 4. Run the OpenIE6 Model
```bash
docker run --gpus all -v $(pwd):/workspace openie6:cuda10 \
  python3 run.py --mode splitpredict \
  --inp sentences.txt --out predictions.txt \
  --task oie --gpus 1 \
  --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt \
  --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt \
  --num_extractions 5
```

### 5. View Results
```bash
cat ./predictions.txt.oie
```

## Expected Output
You should see output similar to:
```
The quick brown fox jumps over the lazy dog .
1.00: (The quick brown fox; jumps; over the lazy dog)

John loves Mary and goes to school .
1.00: (John; loves; Mary)
1.00: (John; goes; to school)

Barack Obama was born in Hawaii .
1.00: (Barack Obama; was born; in Hawaii)
```

# Pipeline Script

The `pipeline.sh` script automates the process of running OpenIE6 on multiple text files. It processes folders numbered 1 to N in the `result/controlled_extraction/test` directory, where N is specified as a command-line argument. For each folder, it:
1. Reads the `text.txt` file
2. Tokenizes the text into sentences using NLTK
3. Runs OpenIE6 to extract relations
4. Saves the predictions (`predictions.txt.oie`) back to the original folder

To run the pipeline:
```bash
./pipeline.sh <number_of_folders>
```

Example:
```bash
./pipeline.sh 90
```
# Processing Predictions

The `process_predictions.py` script processes the OpenIE6 predictions and formats them into a standardized format. It should be run after `pipeline.sh` has completed. The script:
1. Reads `predictions.txt.oie` from each folder (1 to N), where N is the number of folders you processed with the pipeline
2. Converts the format from `(subject; relation; object)` to `["subject", "relation", "object"]`
3. Creates an aggregated file `aggregated_openie6_triplets.txt` in the `original_prediction` folder
4. Preserves Unicode characters in their original form

To process the predictions (after running pipeline.sh), specify the number of folders as an argument:
```bash
python process_predictions.py <number_of_folders>
```

Example:
```bash
python process_predictions.py 90
```

# Filtering OpenIE6 Results for Common Samples

The `filter_common_triplets.py` script filters the OpenIE6 results to only include samples where **all methods produced non-empty results**. This is useful for fair comparison and evaluation.

The script:
1. Reads the aggregated triplet files for each method to determine which samples have non-empty results in all methods.
2. Writes a new `aggregated_openie6_triplets.txt` file (in the parent directory of `original_prediction`) containing only those OpenIE6 samples where all methods produced non-empty results.

**Usage:**
```bash
python filter_common_triplets.py
```

- Make sure the following files exist in `result/controlled_extraction/test/original_prediction/`:
  - `aggregated_chatgpt_triplets.txt`
  - `aggregated_graphrag_triplets.txt`
  - `aggregated_lightrag_triplets.txt`
  - `aggregated_finetuned_1.5B_improved_prediction_triplets.txt`
  - `aggregated_openie6_triplets.txt`
- The filtered results will be saved as `aggregated_openie6_triplets.txt` in `result/controlled_extraction/test/`.
