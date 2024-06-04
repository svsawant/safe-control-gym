#!/bin/bash

# LQR Experiment.

SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_3D'

TASK='stabilization'
# TASK='tracking'

ALGO='linear_mpc'
# ALGO='ilqr'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

python3 ./ROA_cartpole_mpc.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}_ROA.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}_ROA.yaml
