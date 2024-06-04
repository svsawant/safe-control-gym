#!/bin/bash

# LQR Experiment.

SYS='cartpole'
# SYS='quadrotor_2D'
# SYS='quadrotor_3D'

TASK='stabilization'
# TASK='tracking'

ALGO='gp_mpc'


if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

python3 ./test_cartpole_swingup_gp_mpc.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}_${TASK}.yaml
