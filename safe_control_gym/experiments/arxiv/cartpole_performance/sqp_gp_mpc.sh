#!/bin/bash

# MASS='600'
seed='20'
# rm -r -f ./temp-data/

python3 ./utils/gpmpc_cartpole_data_eff.py \
            --algo sqp_gp_mpc --task cartpole \
            --overrides ./config_overrides/sqp_gp_mpc_cartpole.yaml \
            --seed ${seed}

# mv ./temp-data/M_${MASS}/cartpole_data_eff/* ./data/cartpole_data_eff/M_${MASS}/

