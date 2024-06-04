'''Register environments.'''

from safe_control_gym.utils.registration import register

register(idx='cartpole',
         entry_point='safe_control_gym.envs.gym_control.cartpole:CartPole',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole.yaml')

register(idx='quadrotor',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor:Quadrotor',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor.yaml')

register(idx='pendulum',
         entry_point='safe_control_gym.envs.gym_pendulum.pendulum:Pendulum',
         config_entry_point='safe_control_gym.envs.gym_pendulum:pendulum.yaml')

register(idx='shower',
         entry_point='safe_control_gym.envs.test_shower.shower:ShowerEnv',
         config_entry_point='safe_control_gym.envs.test_shower:shower.yaml')