#!/bin/bash

baseline=$1
bert=$2
domain=$3
contra=$4
test_file=$5
evalmode=$6
runs=$7

. ~/anaconda2/etc/profile.d/conda.sh
conda activate p3-torch10


if ! [ -z $8 ] ; then
    export CUDA_VISIBLE_DEVICES=$8
    echo "using cuda"$CUDA_VISIBLE_DEVICES
fi


DATA_DIR="../asc/"$domain

for run in `seq 1 1 $runs`
do
    OUTPUT_DIR="../run/"$baseline/$domain/$run

    mkdir -p $OUTPUT_DIR
    if ! [ -e $OUTPUT_DIR/"valid.json" ] ; then
        python ../src/run_asc.py \
            --bert_model $bert --do_train --do_valid --use_weight $contra\
            --max_seq_length 100 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 4 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --seed $run > $OUTPUT_DIR/train_log.txt 2>&1
    fi

    if ! [ -e $OUTPUT_DIR/"predictions_"${test_file}"_"$evalmode".json" ] ; then 
        python ../src/run_asc.py \
            --bert_model $bert --do_eval --evalmode $evalmode --max_seq_length 100 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --test_file ${test_file}.json --seed $run > $OUTPUT_DIR/test_${test_file}_${evalmode}.log 2>&1
    fi
    if [ -e $OUTPUT_DIR/"predictions_"${test_file}"_"$evalmode".json" ] && [ -e $OUTPUT_DIR/model.pt ] ; then
        rm $OUTPUT_DIR/model.pt
    fi
    
done
