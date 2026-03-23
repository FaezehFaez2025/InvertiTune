#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1

export OUTPUT_DIR_NAME=outputs/t5_large_only_one_error_kelm_new
export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

rm -rf $OUTPUT_DIR

mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=10


export CUDA_VISIBLE_DEVICES=${GPUID}

python ${ROOT_DIR}/finetune.py \
--data_dir=${ROOT_DIR}/data/only_one_error_kelm \
--learning_rate=2e-5 \
--num_train_epochs 5 \
--task graph2text \
--model_name_or_path=${MODEL} \
--train_batch_size=16 \
--eval_batch_size=16 \
--early_stopping_patience 5 \
--gpus 1 \
--output_dir=$OUTPUT_DIR \
--cache_dir=".cache/" \
--max_source_length=512 \
--max_target_length=128 \
--val_max_target_length=128 \
--test_max_target_length=128 \
--eval_max_gen_length=128 \
--do_train --do_predict \
--eval_beams 5 \
--val_check_interval 1.0 \
--val_metric loss \
--save_top_k -1 \
--warmup_steps 500
