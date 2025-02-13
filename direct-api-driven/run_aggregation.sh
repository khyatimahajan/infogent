#!/bin/bash

# Source environment variables
. ./setup_env.sh

# Create results directory
mkdir -p results

# Run FanOutQA aggregation
python run_fanoutqa.py \
    --navigator_model gpt-4o-mini \
    --aggregator_model gpt-4o-mini \
    --extractor_model gpt-4o-mini \
    --chat_deployment ${CHAT_DEPLOYMENT} \
    --embedding_deployment ${EMBEDDING_DEPLOYMENT} \
    --api_version ${API_VERSION} \
    --inp_path data/fanout-final-dev.json \
    --out_path results/fanoutqa_dev_agg.json \
    --log_path results/fanoutqa_dev_agg.log 