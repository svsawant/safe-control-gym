#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_exp'
# SYS='quadrotor_3D'

# TASK='stab'
TASK='track'

# ALGO='ppo'
# ALGO='sac'
# ALGO='td3'
ALGO='ddpg'

EXP_NAME='quad_results'

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
        --tag ${SYS}_${ALGO}_data \
        --seed ${SEED} \
        --use_gpu
        # --kv_overrides \
            # task_config.randomized_init=True 
        # --pretrain_path /home/savvyfox/Projects/scg/examples/rl/models/${ALGO}/ 
done
