#!/bin/bash

python peform_retrieval.py --retrieval_unit "dialogue" --retrieval_type "gt" --output_path "retrieval_results_dialogue_gt.jsonl"
python peform_retrieval.py --retrieval_unit "dialogue" --retrieval_type "asr" --output_path "retrieval_results_dialogue_asr.jsonl"
python peform_retrieval.py --retrieval_unit "user" --retrieval_type "gt" --output_path "retrieval_results_user_gt.jsonl"
python peform_retrieval.py --retrieval_unit "user" --retrieval_type "asr" --output_path "retrieval_results_user_asr.jsonl"
