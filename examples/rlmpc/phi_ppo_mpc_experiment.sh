#!/bin/bash
# phi_ppo_mpc_experiment.sh
#
# USAGE
# -----
#   # Training:
#   bash phi_ppo_mpc_experiment.sh train
#
#   # Evaluation (untrained — uses Phi matrix directly):
#   bash phi_ppo_mpc_experiment.sh eval
#
#   # Evaluation from a checkpoint:
#   bash phi_ppo_mpc_experiment.sh eval ./Results/quadrotor_2D_phi_ppo_mpc_s0/

SYS='quadrotor_2D'
TASK='track'
ALGO='phi_ppo_mpc'
SYS_NAME='quadrotor'

MODE=${1:-eval}          # 'train' or 'eval'
PRETRAIN_PATH=${2:-""}   # optional path to Results/ checkpoint dir

for SEED in {0..0}; do
    echo "============================================================"
    echo "PhiPPO_MPC | SYS=${SYS} | TASK=${TASK} | SEED=${SEED} | MODE=${MODE}"
    echo "============================================================"

    if [ "$MODE" == "train" ]; then
        python3 ../../safe_control_gym/experiments/train_rl_controller.py \
            --algo   ${ALGO} \
            --task   ${SYS_NAME} \
            --overrides \
                ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
                ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
            --output_dir ./experiment_results/ \
            --tag    ${SYS}_${ALGO} \
            --seed   ${SEED}
    else
        PRETRAIN_KV="algo_config.training=False"
        if [ -n "$PRETRAIN_PATH" ]; then
            PRETRAIN_KV="${PRETRAIN_KV} pretrain_path=${PRETRAIN_PATH}"
        fi
        python3 ./rlmpc_experiment.py \
            --task   ${SYS_NAME} \
            --algo   ${ALGO} \
            --seed   ${SEED} \
            --overrides \
                ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
                ./config_overrides/${SYS}/${ALGO}_${SYS}.yaml \
            --kv_overrides ${PRETRAIN_KV}
    fi
done
