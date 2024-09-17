#!/bin/bash

#SYS='cartpole'
#SYS='quadrotor_2D'
SYS='quadrotor_2D_attitude'
#SYS='quadrotor_2D_attitude_5s'
#SYS='quadrotor_3D'

#TASK='stab'
TASK='track'

#ALGO='ppo'
ALGO='sac'
#ALGO='td3'
#ALGO='ddpg'
#ALGO='dppo'
#ALGO='safe_explorer_ppo'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Removed the temporary data used to train the new unsafe model.
# rm -r -f ./${ALGO}_data_2/

#if [ "$ALGO" == 'safe_explorer_ppo' ]; then
#    # Pretrain the unsafe controller/agent.
#    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
#        --algo ${ALGO} \
#        --task ${SYS_NAME} \
#        --overrides \
#            ./config_overrides/${SYS}/${ALGO}_${SYS}_pretrain.yaml \
#            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
#        --output_dir ./unsafe_rl_temp_data/ \
#        --seed 2 \
#        --kv_overrides \
#            task_config.init_state=None
#
#    # Move the newly trained unsafe model.
#    mv ./unsafe_rl_temp_data/model_latest.pt ./models/${ALGO}/${ALGO}_pretrain_${SYS}_${TASK}.pt
#
#    # Removed the temporary data used to train the new unsafe model.
#    rm -r -f ./unsafe_rl_temp_data/
#fi

# Train the unsafe controller/agent.
for SEED in {1..1}
do
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ${ALGO} \
        --task ${SYS_NAME} \
        --overrides \
            ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        --output_dir ./Results/${SYS}_${ALGO}_data/${SEED}/ \
        --seed ${SEED} \
        --use_gpu\
        --kv_overrides \
            task_config.randomized_init=True
        # --pretrain_path ./models/${ALGO}/model_latest.pt
done

# Move the newly trained unsafe model.
#mv ./unsafe_rl_temp_data/model_best.pt ./models/${ALGO}/${ALGO}_model_${SYS}_${TASK}.pt

# Removed the temporary data used to train the new unsafe model.
#rm -r -f ./unsafe_rl_temp_data/
