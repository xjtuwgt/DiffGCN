#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=gatcodes
#The first four parameters must be provided
#Only used in training
SEED=$1
DATANAME=$2
echo "Start Training......"
/mnt/cephfs2/asr/users/ming.tu/sgetools/run_gpu.sh python -u $CODE_PATH/gattrain.py \
    --cuda \
    --shuffle \
    --dataset $DATANAME\
    --rand_seed $SEED