#!/bin/bash

#SYS='cartpole'
#SYS='quadrotor_2D'
SYS='quadrotor_2D_exp'
# SYS='quadrotor_3D'

#TASK='stab'
TASK='track'

ALGO='ppo_mpc'
# ALGO='ppo_vmpc'
# ALGO='appo_mpc'
# ALGO='sac_mpc'
# ALGO='td3_mpc'

EXP_DATA='quad_results'
EVAL_LIST=('basic')
# EVAL_LIST=('performance' 'trajectory_data')

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# RL Experiment
for EVAL in "${EVAL_LIST[@]}"; do
    for SEED in {0..0}; do
        echo "Running experiment with seed ${SEED} and external parameter ${EVAL}"
        python3 ./rlmpc_experiment.py \
            --task ${SYS_NAME} \
            --algo ${ALGO} \
            --seed ${SEED} \
            --overrides \
                ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            --kv_overrides \
                algo_config.training=False \
                task_config.randomized_init=True \
                algo_config.actor_config.mpc_config.jit=False
            # --pretrain_path ./Results/${EXP_DATA}/${SYS}_${ALGO}_data/seed${SEED}_*/ 
    done
    wait
done
