#!/bin/bash

# Source environment variables
. ./setup_env.sh

# Create results directory
mkdir -p results

# Run FanOutQA aggregation
python run_fanoutqa_wiki.py \
    --navigator_model meta-llama/Llama-3.1-70B-Instruct \
    --aggregator_model meta-llama/Llama-3.1-70B-Instruct \
    --extractor_model meta-llama/Llama-3.1-70B-Instruct \
    --chat_deployment ${CHAT_DEPLOYMENT} \
    --embedding_deployment ${EMBEDDING_DEPLOYMENT} \
    --inp_path data/fanout-final-dev.json \
    --out_path results/fanoutqa_dev_agg_3.json \
    --log_path results/fanoutqa_dev_agg_3.log 

echo "Output saved to results/fanoutqa_dev_agg_3.json"
echo "Log saved to results/fanoutqa_dev_agg_3.log"