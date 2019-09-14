#!/bin/bash

mode=$1

#. ~/anaconda2/etc/profile.d/conda.sh
#conda activate p3-torch10

#export CUDA_VISIBLE_DEVICES=1 

export PYTHONPATH="${PYTHONPATH}:./"

if [ $mode == train ] ; then
    python script/build_configs.py
    python script/runner.py
fi

if [ $mode == test ] ; then
    python script/evaluator.py
fi