#!/bin/bash

# SYS='cartpole'
SYS='quadrotor_2D'
# SYS='quadrotor_3D'

# TASK='stab'
TASK='track'

ALGO='ppo'
# ALGO='sac'
# ALGO='safe_explorer_ppo'

EXP_NAME='quad_results3'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
for SEED in {0..0}
do
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./Results/${EXP_NAME}/ \
        --seed ${SEED} \
        --kv_overrides \
            task_config.init_state=None \
            task_config.randomized_init=True 
            # algo_config.pretrained=./models/${ALGO}/${ALGO}_pretrain_${SYS}_${TASK}.pt
done
