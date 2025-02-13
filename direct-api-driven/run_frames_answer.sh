#!/bin/bash

# Source the environment variables
. ./setup_env.sh

# Default values
ANSWER_MODEL=${AZURE_GPT4_DEPLOYMENT:-"gpt-4o-mini"}


# Run the Python script
python frames_answer.py \
    --answer_model "$ANSWER_MODEL" \
    --data_path "data/test_frames_w_ids.json" \
    --inp_path "results/frames_agg.json" \
    --score_path "results/frames_scores_agg.json" \
    --out_path "results/frames_output_agg.json"

echo "Output saved to results/frames_output_agg.json"
echo "Scores saved to results/frames_scores_agg.json" 