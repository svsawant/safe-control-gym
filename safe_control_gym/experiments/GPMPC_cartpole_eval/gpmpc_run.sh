#!/bin/bash

# GP-MPC (with 150 points kernel) for quadrotor environment with diagonal constraint.
python3 ./gpmpc_cartpole.py --task quadrotor --algo gp_mpc --overrides ./config_overrides/gp_mpc_quad.yaml
