#!/bin/bash

baseline=$1
bert=$2
domain=$3
epoch=$4
test_file=$5
runs=$6

evalmode=asp

. ~/anaconda2/etc/profile.d/conda.sh
conda activate p3-torch10


if ! [ -z $7 ] ; then
    export CUDA_VISIBLE_DEVICES=$7
    echo "using cuda"$CUDA_VISIBLE_DEVICES
fi


DATA_DIR="../asc/"$domain

for factor in 0.05
do
    for run in `seq 1 1 $runs`
    do
        OUTPUT_DIR="../run/"$baseline"_"$factor/$domain/$run

        mkdir -p $OUTPUT_DIR
        if ! [ -e $OUTPUT_DIR/"valid.json" ] ; then
            python ../src/run_adaweight_asc.py \
                --bert_model $bert --do_train --do_valid \
                --max_seq_length 100 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs $epoch \
                --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --seed $run --factor $factor > $OUTPUT_DIR/train_log.txt 2>&1
        fi

        test_file=test_full
        if ! [ -e $OUTPUT_DIR/"predictions_"${test_file}"_"$evalmode".json" ] ; then 
        python ../src/run_asc.py \
            --bert_model $bert --do_eval --evalmode $evalmode --max_seq_length 100 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --test_file ${test_file}.json --seed $run > $OUTPUT_DIR/test_${test_file}_${evalmode}.log 2>&1
        fi

        test_file=test_contra
        if ! [ -e $OUTPUT_DIR/"predictions_"${test_file}"_"$evalmode".json" ] ; then 
        python ../src/run_asc.py \
            --bert_model $bert --do_eval --evalmode $evalmode --max_seq_length 100 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --test_file ${test_file}.json --seed $run > $OUTPUT_DIR/test_${test_file}_${evalmode}.log 2>&1
        fi

        if [ -e $OUTPUT_DIR/"predictions_"${test_file}"_"$evalmode".json" ] && [ -e $OUTPUT_DIR/model.pt ] ; then
            rm $OUTPUT_DIR/model.pt
        fi
    done
    
done
