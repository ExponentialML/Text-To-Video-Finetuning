#!/bin/bash

check_points='./outputs/train_2023-11-20T23-10-32/checkpoint-*'
suffix=unique_token_dog_only_$(date +%s)
prompts=("\"A « dog is playing ball\"" "\"A « dog is runnig\"" "\"A dog\"" "\"A dog is running\"")
for prompt in "${prompts[@]}"
do
    per_file_suffix=_"$suffix"_benchmark
    eval "python inference.py --prompt $prompt --model ./models/model_scope_diffusers --sdp --suffix $per_file_suffix --output-dir ./output/$suffix"
    for checkpoint_file in $check_points
    do
        iteration="${checkpoint_file##*-}"
        per_file_suffix=_"$suffix"_"$iteration"

        eval "python inference.py --prompt $prompt --model $checkpoint_file --sdp --suffix $per_file_suffix --output-dir ./output/$suffix"
    done
done