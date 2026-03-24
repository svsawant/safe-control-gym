#!/bin/bash

# SYS='cartpole'
# SYS='quadrotor_2D'
SYS='quadrotor_2D_exp'
# SYS='quadrotor_3D'

# TASK='stab'
TASK='track'

# ALGO='ppo_mpc'
# ALGO='appo_mpc'
ALGO='ppo_vmpc'
# ALGO='sac_mpc'
# ALGO='td3_mpc'

EXP_NAME='test'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
for SEED in {0..0}
do
    echo "Training ${ALGO} on ${SYS_NAME} with seed ${SEED}"
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./Results/${EXP_NAME} \
        --tag ${SYS}_${ALGO} \
        --seed ${SEED} \
        --kv_overrides \
            task_config.randomized_init=True \
            task_config.rew_exponential=False
done
