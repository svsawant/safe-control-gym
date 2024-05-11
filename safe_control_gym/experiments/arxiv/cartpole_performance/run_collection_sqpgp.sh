#!/bin/bash

# MASS='600'
##########################
# prior_param_coeff='1.3'
##########################
# seed='1'
# for prior_param_coeff in '0.7' '1.0' '1.3' '1.5' '3.0'
# for MASS in '0.5'
# do
#     # MASS='2.0'
#     mkdir ./data/cartpole_data_eff/gp_mpc/M_${MASS}
#     for seed in '16' '17' '18' '19' '20' 
#     do 
#         rm -r -f ./temp-data/
#         python3 ./utils/gpmpc_cartpole_data_eff.py \
#                 --algo gp_mpc --task cartpole \
#                 --overrides ./config_overrides/gpmpc_cartpole_M_${MASS}.yaml \
#                 --seed ${seed} 
#         mv ./temp-data/cartpole_data_eff/* \
#     ./data/cartpole_data_eff/gp_mpc/M_${MASS}
#     done
# done


for MASS in '0.5' # '0.1' '1.0' '1.5' '2.0' '3.0'
do
    # MASS='2.0'
    mkdir ./data/cartpole_data_eff/sqp_gp_mpc/M_${MASS}
    # for seed in '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15'
    for seed in '16' '17' '18' '19' '20'
    do 
        rm -r -f ./temp-data/
        python3 ./utils/gpmpc_cartpole_data_eff.py \
                --algo sqp_gp_mpc --task cartpole \
                --overrides ./config_overrides/sqp_gp_mpc_cartpole_M_${MASS}.yaml \
                --seed ${seed} 
        mv ./temp-data/cartpole_data_eff/* \
    ./data/cartpole_data_eff/sqp_gp_mpc/M_${MASS}
    done
done
mv ./temp-data/M_${MASS}/cartpole_data_eff/* ./data/cartpole_data_eff/M_${MASS}/

# for prior_param_coeff in '0.7' '1.0' '1.3' '1.5' '3.0'
# do
#     for seed in '1' '2' '3' '4' '5'
#     do 
#         rm -r -f ./temp-data/
#         python3 ./utils/gpmpc_cartpole_data_eff.py \
#                 --algo sqp_gp_mpc --task cartpole \
#                 --overrides ./config_overrides/sqp_gp_mpc_cartpole.yaml \
#                 --seed ${seed} \
#                 --prior_param_coeff ${prior_param_coeff} 

#         mv ./temp-data/cartpole_data_eff/* \
#     ./data/cartpole_data_eff/sqp_gp_mpc/${prior_param_coeff}/seed${seed}/
#     done
# done

# prior_param_coeff='0.7'
# #########################
# seed='1'
# mv ./temp-data/cartpole_data_eff/* \
#     ./data/cartpole_data_eff/sqp_gp_mpc/${prior_param_coeff}/seed${seed}/