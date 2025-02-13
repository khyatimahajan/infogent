#!/bin/bash

# Source environment variables
. ./setup_env.sh

# Run closed book evaluation
python fanoutqa_answer.py \
    --answer_model gpt-4o --closed_book \
    --chat_deployment ${CHAT_DEPLOYMENT} \
    --embedding_deployment ${EMBEDDING_DEPLOYMENT} \
    --api_version ${API_VERSION} \
    --data_path data/fanout-final-dev.json \
    --out_path results/fanoutqa_dev_closed_answer.json \
    --score_path results/fanoutqa_dev_closed_score.json 