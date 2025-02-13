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
    --inp_path data/test_frames_w_ids.json \
    --out_path results/frames_agg_2.json \
    --log_path results/frames_agg_2.log 