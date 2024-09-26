#!/bin/bash

#SYS='cartpole'
#SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'

#TASK='stab'
TASK='track'

ALGO='qlearning_mpc'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Model-predictive safety certification of an unsafe controller.
python3 ./rlmpc_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml\
        ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml
