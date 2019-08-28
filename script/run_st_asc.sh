#!/bin/bash

baseline=$1
bert=$2
domain=$3
runs=$4
st=$5

. ~/anaconda2/etc/profile.d/conda.sh
conda activate p3-torch10


if ! [ -z $6 ] ; then
    export CUDA_VISIBLE_DEVICES=$6
    echo "using cuda"$CUDA_VISIBLE_DEVICES
fi


ST_DATA_DIR="../asc/"$domain"_st"

for run in `seq 1 1 $st`
do
    OUTPUT_DIR="../run/"$baseline/$domain/$run

    mkdir -p $OUTPUT_DIR
    if ! [ -e $OUTPUT_DIR/"predictions_test_dev_asp.json" ] ; then
        python ../src/run_asc.py \
            --bert_model $bert --do_train \
            --max_seq_length 100 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 2 \
            --output_dir $OUTPUT_DIR --data_dir $ST_DATA_DIR --seed $run > $OUTPUT_DIR/train_log.txt 2>&1
    fi

    test_file=test_train

    if ! [ -e $OUTPUT_DIR/"predictions_"${test_file}"_asp.json" ] ; then 
        python ../src/run_asc.py \
            --bert_model $bert --do_eval --max_seq_length 100 \
            --output_dir $OUTPUT_DIR --data_dir $ST_DATA_DIR --test_file ${test_file}.json --seed $run > $OUTPUT_DIR/test_${test_file}.log 2>&1
    fi
    
    test_file=test_dev
    
    if ! [ -e $OUTPUT_DIR/"predictions_"${test_file}"_asp.json" ] ; then 
        python ../src/run_asc.py \
            --bert_model $bert --do_eval --max_seq_length 100 \
            --output_dir $OUTPUT_DIR --data_dir $ST_DATA_DIR --test_file ${test_file}.json --seed $run > $OUTPUT_DIR/test_${test_file}.log 2>&1
    fi
    
    if [ -e $OUTPUT_DIR/"predictions_"${test_file}"_asp.json" ] && [ -e $OUTPUT_DIR/model.pt ] ; then
        rm $OUTPUT_DIR/model.pt
    fi
done

DATA_DIR="../asc/"$domain"_st_weight"
mkdir -p $DATA_DIR

test_file=test_train

python ../src/build_weight.py \
    --run_dir "../run/"$baseline --domain $domain --run $st --pred_file predictions_${test_file}_asp.json \
    --src_json $ST_DATA_DIR/${test_file}.json --tgt_json $DATA_DIR/train.json > $DATA_DIR/train.log 2>&1

test_file=test_dev

python ../src/build_weight.py \
    --run_dir "../run/"$baseline --domain $domain --run $st --pred_file predictions_${test_file}_asp.json \
    --src_json $ST_DATA_DIR/${test_file}.json --tgt_json $DATA_DIR/dev.json > $DATA_DIR/dev.log 2>&1


########## retraining ###############

baseline=$baseline"_build"

evalmode="asp"

for run in `seq 1 1 $runs`
do
    OUTPUT_DIR="../run/"$baseline/$domain/$run

    mkdir -p $OUTPUT_DIR
    if ! [ -e $OUTPUT_DIR/"valid.json" ] ; then
        python ../src/run_asc.py \
            --bert_model $bert --do_train --do_valid --use_weight 1 \
            --max_seq_length 100 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 4 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --seed $run > $OUTPUT_DIR/train_log.txt 2>&1
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
