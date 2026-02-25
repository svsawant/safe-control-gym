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

EXP_NAME='quad_results'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Train the unsafe controller/agent.
for SEED in {0..4}
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
        --use_gpu \
        # --kv_overrides \
            # 'task_config.rew_state_weight=[1.0,0.1,1.0,0.1,0.01,0.001]'
            # algo_config.opt_epochs=20
            # algo_config.random_process.std.args=0.2
            # task_config.randomized_init=True 
        # --pretrain_path /home/savvyfox/Projects/scg/examples/rl/models/${ALGO}/ 
done
