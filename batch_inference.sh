#!/bin/bash

check_points='./outputs/train_2023-11-14T18-55-08/checkpoint-*'
suffix="unique_token_sks_only"
prompts=("\"A sks person is playing ball.\"" "\"A sks person is playing basketball.\"" "\"A sks person is playing soccer.\"" "\"A sks person is surfing.\"")
for prompt in "${prompts[@]}"
do
    for checkpoint_file in $check_points
    do
        iteration="${checkpoint_file##*-}"
        per_file_suffix=_"$suffix"_"$iteration"

        eval "python inference.py --prompt $prompt --model $checkpoint_file --sdp --suffix $per_file_suffix"
    done
done