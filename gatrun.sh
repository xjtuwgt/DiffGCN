#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=applications
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
DATASET=$2
SAVE_ID=$3
SAVE=$SAVE_PATH/"$DATASET"_"$SAVE_ID"

#Only used in training
LAYERS=$4
HIDDEN_DIM=$5
PROJECT_DIM=$6
HEADS=$7
HOPS=$8
LEARNING_RATE=${9}
FEA_DROP=${10}
ATT_DROP=${11}
NEG_SLOP=${12}
WEIGHTDECAY=${13}
EPOCHS=${14}
ALPHA=${15}
TOPK=${16}
TOPKTYPE=${17}
EDROP=${18}
LOOP=${19}
SEED=${20}


if [ $MODE == "train" ]
then

echo "Start Training......"

/mnt/cephfs2/asr/users/ming.tu/sgetools/run_gpu.sh python -u $CODE_PATH/magantrain.py --do_train \
    --cuda \
    --dataset $DATASET\
    --save_path $SAVE\
    --hop_num $HOPS\
    --epochs $EPOCHS\
    --num_heads $HEADS\
    --num_layers $LAYERS\
    --num_hidden $HIDDEN_DIM\
    --in_drop $FEA_DROP\
    --attn_drop $ATT_DROP\
    --alpha $ALPHA\
    --project_dim $PROJECT_DIM\
    --lr $LEARNING_RATE\
    --negative_slope $NEG_SLOP\
    --top_k $TOPK\
    --weight_decay $WEIGHTDECAY\
    --topk_type $TOPKTYPE\
    --edge_drop $EDROP\
    --self_loop $LOOP\
    --seed $SEED

else
   echo "Unknown MODE" $MODE
fi