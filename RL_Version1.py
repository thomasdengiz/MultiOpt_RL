
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Tuple, MultiDiscrete

import Run_Simulations_Help
import SetUpScenarios
import os
import pickle
import numpy as np
import string
import time
import random



# Record the start time
start_time = time.time()


global help_counter_number_of_0_action
help_counter_number_of_0_action =0
global help_sum_reward_0_action
help_sum_reward_0_action = 0
global help_counter_number_of_1_action
help_counter_number_of_1_action = 0
global help_sum_reward_1_action
help_sum_reward_1_action = 0
global help_counter_number_of_2_action
help_counter_number_of_2_action = 0
global help_sum_reward_2_action
help_sum_reward_2_action = 0


number_of_days_for_training = 12
number_of_new_solutions_per_solution = 20
number_of_new_solutions_per_iteration = 20
number_of_iterations_per_day = 3
use_resulting_state_after_action_as_current_solution = True
string_run_name = "RL1_Days" + str(number_of_days_for_training) + "_SolSol" + str (number_of_new_solutions_per_solution) + "_SolIt" + str(number_of_new_solutions_per_iteration) + "_ItDay" + str (number_of_iterations_per_day) + "_ResState" + str(use_resulting_state_after_action_as_current_solution)

#days_for_training =  [15, 28, 37,  52, 65, 72,   298, 303,310, 328, 346, 352], [18, 31, 32, 49, 74, 80, 290, 302, 305, 331, 349, 345]
days_for_training =  [18, 31, 32, 49, 74, 80, 290, 302, 305, 331, 349, 345]
choose_days_randomly = False

number_of_runs_for_the_algorithm = number_of_days_for_training * number_of_iterations_per_day * number_of_new_solutions_per_iteration * number_of_new_solutions_per_solution
print("Number of runs for the algorithm: " + str(number_of_runs_for_the_algorithm))


class DSM_Env(Env):
    def __init__(self):

        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([3, 3, 3]),  dtype=np.float64)

        #State parameters
        self.state_difference_costs_to_conventional_solution = 0
        self.state_difference_peak_to_conventional_solution = 0
        self.state_thermal_discomfort = 0
        self.help_state_costs_conventional_solution = 0
        self.help_state_peak_conventional_solution = 0
        self.help_state_thermal_discomfort = 0
        self.help_state_resoluting_solution_dicitionary  = 0



        #Auxillary variables
        self.current_number_total_solutions_of_the_run = 0
        self.current_iteration_of_the_training_day = 0
        self.current_training_day = 0
        self.solution_of_current_file = 0
        self.read_data_for_new_iteration = True
        self.list_of_solution_current_training_day = []
        self.help_counter_current_number_new_solution_per_solution = 0
        self.help_counter_solution_total = 0
        self.help_counter_iteration_current_day =0
        self.help_index_current_day = 0

        self.read_RL_data_day = -1
        self.read_RL_data_iteration = 0

        if choose_days_randomly == True:
            random_index = random.randint(0, len(days_for_training) - 1)
            chosen_day = days_for_training[random_index]
            self.read_RL_data_day = chosen_day
            print(f"-------New Day {chosen_day}------------")
        else:
            chosen_day = days_for_training [self.help_index_current_day]
            self.read_RL_data_day = chosen_day
            print(f"-------New Day {chosen_day}------------")

    def reset (self, **kwargs):

        print(f"Reselt called")
        #Read the base solution when a new training day is used
        file_path = r"C:\Users\wi9632\bwSyncShare\Eigene Arbeit\Code\Python\Demand_Side_Management\MultiOpt_RL\RL\RL_Input\list_population_NB" + str(SetUpScenarios.numberOfBuildings_Total) + "_Day" + str(self.read_RL_data_day) + "_It" + str(self.read_RL_data_iteration) + ".pkl"
        # Load the list from the file
        try:
            with open(file_path, "rb") as file:
                self.list_of_solution_current_training_day = pickle.load(file)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return 0, 0, 0

        #read conventional solution
        file_path = r"C:\Users\wi9632\bwSyncShare\Eigene Arbeit\Code\Python\Demand_Side_Management\MultiOpt_RL\RL\RL_Input\list_population_NB" + str(SetUpScenarios.numberOfBuildings_Total) + "_Day" + str(self.read_RL_data_day) + "_It" + str(0) + ".pkl"
        try:
            with open(file_path, "rb") as file:
                conventional_solutions = pickle.load(file)

            self.help_state_costs_conventional_solution = conventional_solutions[0]['simulationObjective_costs_Euro_combined'][0]
            self.help_state_peak_conventional_solution = conventional_solutions[0] ['simulationObjective_maximumLoad_kW_combined'][0]
            self.help_state_thermal_discomfort = conventional_solutions[0]['simulationObjective_thermalDiscomfort_combined'][0]
            print(f"self.help_state_costs_conventional_solution: {self.help_state_costs_conventional_solution}")

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return 0, 0, 0

        self.read_data_for_new_iteration = False


        super().reset(**kwargs)
        self.state_difference_costs_to_conventional_solution = 0
        self.state_difference_peak_to_conventional_solution = 0
        self.state_thermal_discomfort = 0
        info = {}

        # Execute dummy action in the external simulation and return the next observation, reward, done, and info
        current_solution = self.list_of_solution_current_training_day[self.solution_of_current_file]
        result_costs, result_peak, result_DC, results_dict, result_costs_before_action, result_peak_before_action, result_DC_before_action= Run_Simulations_Help.execute_single_modification_operator_decision_RL1(current_solution, 0, self.read_RL_data_iteration, self.read_RL_data_day )

        #calculate state
        self.state_thermal_discomfort = round(result_DC[0], 3)
        self.state_difference_peak_to_conventional_solution = round(result_peak[0] /self.help_state_peak_conventional_solution, 3)
        self.state_difference_costs_to_conventional_solution= round(result_costs[0] /self.help_state_costs_conventional_solution, 3)
        observation_space = np.array([self.state_difference_costs_to_conventional_solution, self.state_difference_peak_to_conventional_solution,self.state_thermal_discomfort])
        observation_space= observation_space.flatten()

        self.help_state_resoluting_solution_dicitionary = 0

        return  observation_space, info


    def render (self):

        pass

    def step(self, action ):
        global help_counter_number_of_0_action
        global help_sum_reward_0_action
        global help_counter_number_of_1_action
        global help_sum_reward_1_action
        global help_counter_number_of_2_action
        global help_sum_reward_2_action



        current_solution = self.list_of_solution_current_training_day[self.solution_of_current_file]

        #Change current solution if desired
        if use_resulting_state_after_action_as_current_solution == True and self.help_state_resoluting_solution_dicitionary !=0:
            current_solution = self.help_state_resoluting_solution_dicitionary

        peak_load_current_solution = current_solution['simulationObjective_maximumLoad_kW_combined'][0]
        costs_current_solution = current_solution['simulationObjective_costs_Euro_combined'][0]
        thermal_discomfort_current_solution = current_solution['simulationObjective_thermalDiscomfort_combined'][0]


        print(f"current_iteration_overall: {self.help_counter_solution_total}")

        # Execute the action in the external simulation and return the next observation, reward, done, and info
        result_costs, result_peak, result_DC, results_dict, result_costs_before_action, result_peak_before_action, result_DC_before_action = Run_Simulations_Help.execute_single_modification_operator_decision_RL1(current_solution, action, self.read_RL_data_iteration, self.read_RL_data_day )

        #calculate state
        self.state_thermal_discomfort = round(result_DC[0], 3)
        self.state_difference_peak_to_conventional_solution = round(result_peak[0] /self.help_state_peak_conventional_solution, 3)
        self.state_difference_costs_to_conventional_solution= round(result_costs[0] /self.help_state_costs_conventional_solution, 3)
        observation_space = np.array([self.state_difference_costs_to_conventional_solution, self.state_difference_peak_to_conventional_solution,self.state_thermal_discomfort])
        observation_space= observation_space.flatten()
        print(f"action: {action}")
        print(f"observation_space: {observation_space}")


        self.help_counter_solution_total +=1
        self.help_state_resoluting_solution_dicitionary = results_dict


        #Calculate reward
        improvement_peak = (1 - result_peak [0]/ peak_load_current_solution) * 10
        improvement_costs = (1 - result_costs[0]/ costs_current_solution) * 10
        improvement_discomfort = 1 - result_DC[0]/thermal_discomfort_current_solution
        if improvement_discomfort <0 and result_DC[0] <Run_Simulations_Help.threshold_discomfort_local_search:
            improvement_discomfort = improvement_discomfort * 0.5

        if thermal_discomfort_current_solution < Run_Simulations_Help.threshold_discomfort_local_search and result_DC[0] >Run_Simulations_Help.threshold_discomfort_local_search:
            improvement_discomfort = improvement_discomfort * 3

        if thermal_discomfort_current_solution > Run_Simulations_Help.threshold_discomfort_local_search and result_DC[0] <Run_Simulations_Help.threshold_discomfort_local_search:
            improvement_discomfort = improvement_discomfort * 4

        if result_DC[0] >Run_Simulations_Help.threshold_discomfort_local_search + 0.1:
            improvement_discomfort = improvement_discomfort * 2
            done =True


        #Adjust the reward
        if action == 0 and improvement_costs >0:
            improvement_costs = improvement_costs * 8 + (costs_current_solution - result_costs[0]) * 1
            if improvement_peak >=0 and result_DC[0] < Run_Simulations_Help.threshold_discomfort_local_search:
                improvement_costs = improvement_costs * 8
        if action == 1 and improvement_peak >0:
            improvement_peak = improvement_peak * 2
        if action == 2:
            improvement_discomfort = improvement_discomfort * 2

        reward = improvement_peak + improvement_costs + improvement_discomfort
        #Calculate statistics for the different actions
        if action == 0:
            help_counter_number_of_0_action = help_counter_number_of_0_action + 1
            help_sum_reward_0_action += reward
        if action == 1:
            help_counter_number_of_1_action += 1
            help_sum_reward_1_action += reward
        if action == 2:
            help_counter_number_of_2_action += 1
            help_sum_reward_2_action += reward





        print(f"cost_previous: {round(costs_current_solution,2)}")
        print(f"cost_current: {round(result_costs[0],2)}")
        print(f"cost_improvement: {round(improvement_costs,2)}")
        print(f"peak_previous: {round(peak_load_current_solution,2)}")
        print(f"peak_current: {round(result_peak[0],2)}")
        print(f"peak_improvement: {round(improvement_peak,2)}")
        print(f"DC_previous: {round(thermal_discomfort_current_solution,2)}")
        print(f"DC_current: {round(result_DC[0],2)}")
        print(f"DC_improvement: {round(improvement_discomfort,2)}")


        print(f"Reward: {round(reward,2)}")
        done = False

        #Check termination criteria


        #Update auxilliary variables and check terminiation conditions
        self.help_counter_current_number_new_solution_per_solution+= 1
        if self.help_counter_current_number_new_solution_per_solution >= number_of_new_solutions_per_solution:
            # Change the read base solution by choosing the next base solution from the same iteration
            self.solution_of_current_file += 1
            self.help_counter_current_number_new_solution_per_solution = 0
            done = True
        if self.solution_of_current_file >= len(self.list_of_solution_current_training_day) or self.solution_of_current_file >= number_of_new_solutions_per_iteration:
            #Change the read base solution by choosing the base solution from a new iteration
            self.solution_of_current_file = 0
            self.help_counter_current_number_new_solution_per_solution = 0
            self.read_RL_data_iteration += 1
            self.help_counter_iteration_current_day += 1
            done = True
        if result_DC[0] >Run_Simulations_Help.threshold_discomfort_local_search + 0.1:
            #Punish discomfort violations strongly
            improvement_discomfort = improvement_discomfort * 7
            done =True
        if  self.help_counter_iteration_current_day > number_of_iterations_per_day:
            #Change the read base solution by choosing the base solution from a new day
            if choose_days_randomly == True:
                random_index = random.randint(0, len(days_for_training) - 1)
                chosen_day = days_for_training[random_index]
            else:
                self.help_index_current_day+=1
                try:
                    chosen_day = days_for_training[self.help_index_current_day]
                except:
                    self.help_index_current_day = 0
                    chosen_day = days_for_training[self.help_index_current_day]
                    done = True
            self.read_RL_data_day = chosen_day
            self.solution_of_current_file = 0
            self.read_RL_data_iteration = 0
            self.help_counter_current_number_new_solution_per_solution = 0
            self.help_counter_iteration_current_day = 0
            print(f"-------New Day {chosen_day}------------")


        info = {}
        print("")
        return observation_space, reward, done, False, info


#Use Stable Baselines 3 to apply a RL algorithm (ppo, A2C, TD3, dqn) on the environmetn
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

import stable_baselines3 as sb3

# Define a callback to log training information
class CustomCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "training.log")
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Log custom training information to the text file
        with open(self.log_file, "a") as f:
            f.write(self._log())
        return True

    def _log(self):
        time_elapsed = time.time() - self.start_time
        # Access custom information from your environment and log it here
        log_string = f"time/                   |             |\n"
        log_string += f"iterations           | {self.num_timesteps}          |\n"
        log_string += f"time_elapsed         | {time_elapsed:.2f}      |\n"
        log_string += f"total_timesteps      | {self.num_timesteps}          |\n"
        # Add custom information here
        log_string += "-----------------------------------------\n"
        return log_string


#env = DSM_Env(total_number_of_solutions_per_day, number_of_days_for_training, number_of_new_solutions_per_solution)
gym.register("dsm-env-v0", lambda: DSM_Env())
env = gym.make("dsm-env-v0")
#Ceck environment
check_environment = False
if check_environment == True:
    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped)
    from stable_baselines3.common.env_checker import check_env
    check_env(env)




#Create the files of the model
characters = string.ascii_letters  # Includes uppercase and lowercase letters
random_string = random.choice(characters) + random.choice(characters)

#Define the model directory (PPO, A2C, TD3, DQN)
models_dir = r"C:\Users\wi9632\bwSyncShare\Eigene Arbeit\Code\Python\Demand_Side_Management\MultiOpt_RL\RL\RL_Models\\" + string_run_name + "_DQN_" + random_string
logdir = r"C:\Users\wi9632\bwSyncShare\Eigene Arbeit\Code\Python\Demand_Side_Management\MultiOpt_RL\RL\RL_Logs\\" + string_run_name + "_DQN_" + random_string
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

#Define the model (PPO, A2C, TD3, DQN)
#model = A2C('MlpPolicy', env, verbose=1, ent_coef=0.01)
model = DQN('MlpPolicy', env, verbose=1)
callback = CustomCallback(logdir)  # Specify the log directory

#train and save the model
model.learn(total_timesteps=number_of_runs_for_the_algorithm - 2, callback=callback)
model.save(os.path.join(models_dir, 'trained_DQN_model'))


# Calculate the elapsed time in seconds
end_time = time.time()
elapsed_time = end_time - start_time

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print("")
print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")
print("")
percentage_0_action = round(((help_counter_number_of_0_action / number_of_runs_for_the_algorithm) * 100),2 )
percentage_1_action = round(((help_counter_number_of_1_action / number_of_runs_for_the_algorithm) * 100),2 )
percentage_2_action = round(((help_counter_number_of_2_action / number_of_runs_for_the_algorithm) * 100),2 )

average_reward_0_action = round(((help_sum_reward_0_action / help_counter_number_of_0_action)),2 )
average_reward_1_action = round(((help_sum_reward_1_action / help_counter_number_of_1_action)),2 )
average_reward_2_action = round(((help_sum_reward_2_action / help_counter_number_of_2_action)),2 )

print(f"number_of_runs_for_the_algorithm: {number_of_runs_for_the_algorithm}")
print(f"percentage_0_action: {percentage_0_action}")
print(f"percentage_1_action: {percentage_1_action}")
print(f"percentage_2_action: {percentage_2_action}")

print(f"average_reward_0_action: {average_reward_0_action}")
print(f"average_reward_1_action: {average_reward_1_action}")
print(f"average_reward_2_action: {average_reward_2_action}")

