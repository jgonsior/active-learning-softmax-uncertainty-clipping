#!/bin/sh
NUM_ITER=3
BATCH_SIZE=25
EXP_NAME=lunchtest
RANDOM_SEED=42

for UNCERTAINTY_METHOD in softmax temp_scaling label_smoothing MonteCarlo inhibited evidential1; do
    for QUERY_STRATEGY in LC; do
        for RANDOM_SEED in 42; do
            for DATASET in trec6 ag_news subj rotten imdb; do
                python test.py --num_iterations $NUM_ITER --batch_size $BATCH_SIZE --exp_name $EXP_NAME --dataset $DATASET --random_seed $RANDOM_SEED --query_strategy $QUERY_STRATEGY --uncertainty_method $UNCERTAINTY_METHOD
            done
        done
    done
done