#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_exp'
# SYS='quadrotor_3D'

# TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='sac'
# ALGO='td3'
# ALGO='ddpg'

EXP_DATA='quad_results'
EVAL_LIST=('performance' 'trajectory_data')

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# RL Experiment
for EVAL in "${EVAL_LIST[@]}"; do
    for SEED in {0..4}; do
        echo "Running experiment with seed ${SEED} and external parameter ${EVAL}"
        python3 ./rl_experiment.py \
            --task ${SYS_NAME} \
            --algo ${ALGO} \
            --overrides \
                ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            --experiment_type ${EVAL} \
            --seed ${SEED} \
            --kv_overrides \
                algo_config.training=False \
                task_config.randomized_init=True \
            --pretrain_path ./Results/${EXP_DATA}/${SYS}_${ALGO}_data/seed${SEED}_*/ &
    done
    wait
done
