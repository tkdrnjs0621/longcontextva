#!/bin/bash

trap "echo 'SIGINT received. Killing all...'; kill 0; exit 1" SIGINT

CUDA_VISIBLE_DEVICES=3 python inference.py \
    --data_path /home/tkdrnjs0621/work/lcva/retrieval_results/retrieval_results_dialogue_asr.jsonl \
    --prompt_type v1 \
    --audio_dir /home/tkdrnjs0621/work/lcva/dataset/recall_wavs \
    --output_path /home/tkdrnjs0621/work/lcva/outputs/dialogue_asr/v1 &

# CUDA_VISIBLE_DEVICES=4 python inference.py \
#     --data_path /home/tkdrnjs0621/work/lcva/retrieval_results/retrieval_results_dialogue_gt.jsonl \
#     --prompt_type v1 \
#     --output_path /home/tkdrnjs0621/work/lcva/outputs/dialogue_gt/v1 &

# CUDA_VISIBLE_DEVICES=5 python inference.py \
#     --data_path /home/tkdrnjs0621/work/lcva/retrieval_results/retrieval_results_user_asr.jsonl \
#     --prompt_type v1 \
#     --output_path /home/tkdrnjs0621/work/lcva/outputs/user_asr/v1 &

# CUDA_VISIBLE_DEVICES=6 python inference.py \
#     --data_path /home/tkdrnjs0621/work/lcva/retrieval_results/retrieval_results_user_gt.jsonl \
#     --prompt_type v1 \
#     --output_path /home/tkdrnjs0621/work/lcva/outputs/user_gt/v1 
    