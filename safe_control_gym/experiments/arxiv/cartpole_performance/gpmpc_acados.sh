#!/bin/bash

# MASS='600'
seed='4'
# rm -r -f ./temp-data/

python3 ./utils/gpmpc_cartpole_data_eff.py \
            --algo gpmpc_acados --task cartpole \
            --overrides ./config_overrides/gpmpc_acados_cartpole_stabilization.yaml \
            --seed ${seed}

# mv ./temp-data/M_${MASS}/cartpole_data_eff/* ./data/cartpole_data_eff/M_${MASS}/

