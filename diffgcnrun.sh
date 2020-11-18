#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
#The first four parameters must be provided
#Only used in training
SEED=$1
echo "Start Training......"
/mnt/cephfs2/asr/users/ming.tu/sgetools/run_gpu.sh python -u $CODE_PATH/example.py \
    --cuda \
    --shuffle \
    --rand_seed $SEED