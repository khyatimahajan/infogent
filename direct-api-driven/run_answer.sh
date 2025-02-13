#!/bin/bash

# Source environment variables
. ./setup_env.sh

# Set FanoutQA OpenAI API key from Azure settings
export FANOUTQA_OPENAI_API_KEY=${AZURE_OPENAI_KEY}

# Run answer generation and evaluation
python fanoutqa_answer.py \
    --answer_model gpt-4o-mini \
    --chat_deployment ${CHAT_DEPLOYMENT} \
    --api_version ${API_VERSION} \
    --data_path data/fanout-final-dev.json \
    --inp_path results/fanoutqa_dev_agg.json \
    --out_path results/fanoutqa_dev_answer.json \
    --score_path results/fanoutqa_dev_score.json 