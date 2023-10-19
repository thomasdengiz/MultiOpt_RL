
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Tuple, MultiDiscrete, space
import numpy as np
import os




class DSM_Env_RL2(Env):
    def __init__(self):

        # Define the bounds for each dimension of the action space
        action_space_bounds = [10, 10,51]

        # Create a MultiDiscrete action space
        self.action_space = MultiDiscrete(action_space_bounds)

        #Specify observation space
        low = np.zeros(2 * 5)
        high = np.ones(2 * 5)

        # Create the observation space
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        self.observation_space = observation_space

    def reset (self, **kwargs):
        super().reset(**kwargs)
        # Specify observation space
        low = np.zeros(2 * 5)
        high = np.ones(2 * 5)
        info = {}
        # Create the observation space
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        self.observation_space = observation_space

        return low, info

    def render (self):

        pass

    def step(self, action ):

        # Execute the action in the external simulation and return the next observation, reward, done, and info
        action_from_timeslot = action[0]
        action_to_timeslot = action[1]
        action_shifting_percentage = action[2]

        #External environment is not used in this test example
        #result_costs, result_peak, result_DC, results_dict, percentage_array_loads_per_timeslot_highest_prices_shortened, percentage_array_loads_per_timeslot_lowest_prices_shortened = Run_Simulations_Help.execute_single_modification_operator_decision_RL2(current_solution, action_from_timeslot, action_to_timeslot, action_shifting_percentage, self.read_RL_data_day, timeslots_for_state_load_percentages_costs )

        percentage_array_loads_per_timeslot_highest_prices_shortened =np.zeros(5)
        percentage_array_loads_per_timeslot_lowest_prices_shortened  =np.zeros(5)
        #calculate state
        state_array = np.concatenate((percentage_array_loads_per_timeslot_highest_prices_shortened, percentage_array_loads_per_timeslot_lowest_prices_shortened))
        observation_space= state_array

        reward = 1
        done = False

        info = {}
        print("")
        return observation_space, reward, done, False, info


#Use Stable Baselines 3 to apply a RL algorithm on the environmetn
from stable_baselines3 import A2C

gym.register("dsm-env-v1", lambda: DSM_Env_RL2())
env = gym.make("dsm-env-v1")

#Ceck environment
check_environment = False
if check_environment == True:
    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped)
    from stable_baselines3.common.env_checker import check_env
    check_env(env)

#Create the files of the model
models_dir = r"C:\Users\wi9632\Desktop\Ergebnisse\DSM\RL\RL_Models\A2C"
logdir = r"C:\Users\wi9632\Desktop\Ergebnisse\DSM\RL\RL_Logs\A2C"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

#Define the model
model = A2C('MlpPolicy', env, verbose=1)

#train and save the model
model.learn(total_timesteps=1000)
model.save(os.path.join(models_dir, 'trained_a2c_model'))

#########
