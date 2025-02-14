#!/bin/bash

# Source the environment variables
. ./setup_env.sh

# # Default values
# ANSWER_MODEL=${AZURE_GPT4_DEPLOYMENT:-"gpt-4o-mini"}


# Run the Python script
python frames_answer.py \
    --answer_model "meta-llama/Llama-3.1-70B-Instruct" \
    --data_path "data/test_frames_w_ids.json" \
    --inp_path "results/frames_agg_3.json" \
    --score_path "results/frames_scores_agg_3.json" \
    --out_path "results/frames_output_agg_3.json"

echo "Output saved to results/frames_output_agg_3.json"
echo "Scores saved to results/frames_scores_agg_3.json" 