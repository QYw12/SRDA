#!/bin/bash

# custom config
DATA=# ********** your directory ***********
DATASET=$1
CFG=$2  # config file
TRAINER=$3
BACKBONE=$4 # backbone name
NTOK=$5
DOMAINS=$6
GPU=$7

LOCATION=middle
DEEPLAYER=None
TP=True

# text prompt
# TDEEP=False
# VP=False
# VDEEP=False
# SHARE=False

# multi-modal prompt
TDEEP=True
VP=True
VDEEP=True
SHARE=True

TEXT_LOSS_WEIGHT=15
IMAGE_LOSS_WEIGHT=4
GPA_MEAN=6
GPA_STD=10
Q=10
W_S=0.95
W_LOW=0.7
W_HIGH=0.99
ALPHA=0.8

DIR=output/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/tdeep${TDEEP}_vdeep${VDEEP}_${LOCATION}/${DOMAINS}_ntok${NTOK}

if [ -d "$DIR" ]; then
  echo "Results are available in ${DIR}, so skip this job"
else
  echo "Run this job and save the output to ${DIR}"
  
  python train.py \
    --gpu ${GPU} \
    --backbone ${BACKBONE} \
    --domains ${DOMAINS} \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/SRDA/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.SRDA.TP ${TP}\
    TRAINER.SRDA.T_DEEP ${TDEEP} \
    TRAINER.SRDA.N_CTX ${NTOK} \
    TRAINER.SRDA.VP ${VP} \
    TRAINER.SRDA.V_DEEP ${VDEEP}\
    TRAINER.SRDA.NUM_TOKENS ${NTOK} \
    TRAINER.SRDA.LOCATION ${LOCATION} \
    TRAINER.SRDA.DEEP_LAYERS ${DEEPLAYER} \
    TRAINER.SRDA.DEEP_SHARED ${SHARE} \
    TRAINER.SRDA.DDCCR.TEXT_LOSS_WEIGHT ${TEXT_LOSS_WEIGHT}\
    TRAINER.SRDA.DDCCR.IMAGE_LOSS_WEIGHT  ${IMAGE_LOSS_WEIGHT}\
    TRAINER.SRDA.DDCCR.GPA_MEAN ${GPA_MEAN}\
    TRAINER.SRDA.DDCCR.GPA_STD ${GPA_STD}\
    TRAINER.SRDA.Q ${Q}\
    TRAINER.SRDA.W_S ${W_S}\
    TRAINER.SRDA.W_LOW ${W_LOW}\
    TRAINER.SRDA.W_HIGH ${W_HIGH}\
    TRAINER.SRDA.ALPHA ${ALPHA}


    
fi
