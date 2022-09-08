#!/bin/sh
NUM_ITER=20
BATCH_SIZE=25
EXP_NAME=baseline
RANDOM_SEED=42

for UNCERTAINTY_METHOD in softmax temp_scaling label_smoothing MonteCarlo inhibited evidential1 evidential2 bayesian ensembles trustscore model_calibration; do
    for QUERY_STRATEGY in LC Rand Ent MM; do
        for RANDOM_SEED in 42 43 44 45 46; do
            for DATASET in trec6 ag_news subj rotten imdb; do
                python test.py --num_iterations $NUM_ITER --batch_size $BATCH_SIZE --exp_name $EXP_NAME --dataset $DATASET --random_seed $RANDOM_SEED --query_strategy $QUERY_STRATEGY --uncertainty_method $UNCERTAINTY_METHOD
            done
        done
    done
done