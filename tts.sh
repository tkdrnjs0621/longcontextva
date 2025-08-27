#!/bin/bash

trap "echo '[INFO] Caught SIGINT, terminating...'; kill 0" SIGINT

NUM_SHARDS=4

for i in $(seq 0 $((NUM_SHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$i python /home/tkdrnjs0621/work/longcontextva/src/tts_multiple.py \
        --input_file "$INPUT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --device cuda \
        --n 5 \
        --num_processes 4 \
        --shard_id $i \
        --num_shards $NUM_SHARDS &
done

wait
