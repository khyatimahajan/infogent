#!/bin/bash

# Source environment variables
. ./setup_env.sh

# Set FanoutQA OpenAI API key from Azure settings
export FANOUTQA_OPENAI_API_KEY=${AZURE_OPENAI_KEY}

# Run answer generation and evaluation
python fanoutqa_answer.py \
    --answer_model meta-llama/Llama-3.1-70B-Instruct \
    --chat_deployment ${CHAT_DEPLOYMENT} \
    --data_path data/fanout-final-dev.json \
    --inp_path results/fanoutqa_dev_agg.json \
    --out_path results/fanoutqa_dev_answer_2.json \
    --score_path results/fanoutqa_dev_score_2.json 

echo "Output saved to results/fanoutqa_dev_answer_3.json"
echo "Scores saved to results/fanoutqa_dev_score_3.json"