#!/bin/bash


cd /root/autodl-tmp/HybridNER
DATA_DIR="/root/autodl-tmp/HybridNER/dataset/conll03"
BERT_CONFIG_DIR="/root/autodl-tmp/HybridNER/models/bert-base-cased"
MODEL_SAVE_DIR="/root/autodl-tmp/HybridNER/output"

python main.py \
    --loss 'ce' \
    --test_mode 'ori' \
    --etrans_func "softmax" \
    --data_dir "${DATA_DIR}" \
    --state "train" \
    --bert_config_dir "${BERT_CONFIG_DIR}" \
    --batch_size 64 \
    --max_spanLen 5 \
    --n_class 5 \
    --model_save_dir "${MODEL_SAVE_DIR}" \
    --iteration 100