#!/bin/bash

########################
# CONTEXT CURRICULUM
########################
CURRICULUM="5 5 5 5 5   0 0 0 0 0   0 0 0 0 0   0 0 0 0 0   0 0 0 0 0   0 0 0 0 0   0 0 0 0 0   0 0 0 0 0 "
#CURRICULUM="2 2 2 2 2   2 2 2 2 2   2 2 2 2 2   2 2 2 2 2   2 2 2 2 2   2 2 2 2 2   2 2 2 2 2   2 2 2 2 2  --allow-empty-context "
#CURRICULUM="3 3 3 3 3   3 3 3 3 3   3 3 3 3 3   3 3 3 3 3   3 3 3 3 3   3 3 3 3 3   3 3 3 3 3   3 3 3 3 3  --allow-empty-context "


########################
# DATASET
########################

DATASET="colorshapes --query-size 2 --nz 4 --num-views 8 --nchannels 3 --recon-loss scalar_gaussian --num-draw-steps 6 \
--num-actions 0 --cache exp_colorshapes --lr 0.00001 "


#########
# MODEL
#########

MODEL="tgqn-pd --nc-enc 128 --nc-lstm 128 --nc-context 256 --context-type backward --sssm-num-state 108 --shared-core --concatenate-latents --use-ssm-context \
--q-bernoulli-pick 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 --vis-template vis_templates/snp_pd_template.json"

#MODEL="tgqn --nc-enc 128 --nc-lstm 128 --nc-context 256 --context-type backward --sssm-num-state 108 --shared-core --concatenate-latents --use-ssm-context \
#--vis-template vis_templates/snp_template.json "

#MODEL="gqn --nc-enc 128 --nc-lstm 128 --nc-context 256 --shared-core --concatenate-latents \
#--vis-template vis_templates/gqn_template.json "


##################
# GLOBAL SETTINGS
##################
BATCH_SIZE=4
NUM_EPOCHS=5000


########
# LAUNCH
########

python train.py \
--dataset ${DATASET} --nheight 64 \
--model ${MODEL} --state-dict \
--manual-curriculum-bound ${CURRICULUM} \
--train-batch-size ${BATCH_SIZE} --eval-batch-size 4 \
--num-timesteps 10 \
--cuda-device 0 1 2 3  --clip 10 --vis-interval 50 --epochs ${NUM_EPOCHS}

