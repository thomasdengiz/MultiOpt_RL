
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


#Parameters of the iterations
number_of_days_for_training = 12
number_of_new_solutions_per_solution = 10
number_of_new_solutions_per_iteration = 10
number_of_iterations_per_day = 2
use_resulting_state_after_action_as_current_solution = True

#days_for_testing =  [15, 28, 37,  52, 65, 72,   298, 303,310, 328, 346, 352]
days_for_testing =  [15, 28, 37,  52, 65, 72,   298, 303,310, 328, 346, 352]
choose_days_randomly = False

number_of_runs_for_the_algorithm = number_of_days_for_training * number_of_iterations_per_day * number_of_new_solutions_per_iteration * number_of_new_solutions_per_solution
print("Number of runs for the algorithm: " + str(number_of_runs_for_the_algorithm))

#Parameters of the agent (action and state space)
timeslots_for_state_load_percentages_costs = 4
number_of_discrete_shifting_actions = 20
minimum_shifting_percentage = 20
maximum_shifting_percentage = 40


string_run_name = "RL2_Days" + str(number_of_days_for_training) + "_SolSol" + str (number_of_new_solutions_per_solution) + "_SolIt" + str(number_of_new_solutions_per_iteration) + "_ItDay" + str (number_of_iterations_per_day) + "_ResState" + str(use_resulting_state_after_action_as_current_solution) + "_StateTimeSlots" + str(timeslots_for_state_load_percentages_costs) + "_ShiftActions" +  str(number_of_discrete_shifting_actions)


class DSM_Env(Env):
    def __init__(self):


        # Define the bounds for each dimension of the action space
        action_space_bounds = [timeslots_for_state_load_percentages_costs, timeslots_for_state_load_percentages_costs,number_of_discrete_shifting_actions + 1]

        # Create a MultiDiscrete action space
        self.action_space = MultiDiscrete(action_space_bounds)


        #Specify observation space
        low = np.zeros(2 * timeslots_for_state_load_percentages_costs)
        high = np.full(2 * timeslots_for_state_load_percentages_costs, 100.0, dtype=np.float64)

        # Create the observation space
        observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        self.observation_space = observation_space

        #State parameters
        self.state_difference_costs_to_conventional_solution = 0
        self.state_difference_peak_to_conventional_solution = 0
        self.state_thermal_discomfort = 0
        self.help_state_costs_conventional_solution = 0
        self.help_state_peak_conventional_solution = 0
        self.help_state_thermal_discomfort = 0
        self.help_state_resoluting_solution_dicitionary  = 0
        self.percentage_array_loads_per_timeslot_highest_prices_base_solution = 0
        self.percentage_array_loads_per_timeslot_lowest_prices_base_solution = 0

        #Initialize first state
        low = np.zeros(timeslots_for_state_load_percentages_costs)
        high = np.ones(timeslots_for_state_load_percentages_costs)
        self.percentage_array_loads_per_timeslot_highest_prices_base_solution = low
        self.percentage_array_loads_per_timeslot_highest_prices_base_solution = high



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
            random_index = random.randint(0, len(days_for_testing) - 1)
            chosen_day = days_for_testing[random_index]
            self.read_RL_data_day = chosen_day
            print(f"-------New Day {chosen_day}------------")
        else:
            chosen_day = days_for_testing [self.help_index_current_day]
            self.read_RL_data_day = chosen_day
            print(f"-------New Day {chosen_day}------------")

    def reset (self, **kwargs):

        print(f"Reselt called")
        #Read the base solution when a new training day or iteration is used
        file_path = r"C:\Users\wi9632\Desktop\Ergebnisse\DSM\RL\RL_Input\list_population_NB" + str(SetUpScenarios.numberOfBuildings_Total) + "_Day" + str(self.read_RL_data_day) + "_It" + str(self.read_RL_data_iteration) + ".pkl"
        # Load the list from the file
        try:
            with open(file_path, "rb") as file:
                self.list_of_solution_current_training_day = pickle.load(file)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return 0, 0, 0


        #read conventional solution
        file_path = r"C:\Users\wi9632\Desktop\Ergebnisse\DSM\RL\RL_Input\list_population_NB" + str(SetUpScenarios.numberOfBuildings_Total) + "_Day" + str(self.read_RL_data_day) + "_It" + str(0) + ".pkl"
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


        #Call the super method
        super().reset(**kwargs)

        #Exectue one "dummy" action in the environment to just get the initial state space
        current_solution = self.list_of_solution_current_training_day[self.solution_of_current_file]
        result_costs, result_peak, result_DC, results_dict, percentage_array_loads_per_timeslot_highest_prices_shortened, percentage_array_loads_per_timeslot_lowest_prices_shortened, percentage_array_loads_per_timeslot_highest_prices_shortened_before_action, percentage_array_loads_per_timeslot_lowest_prices_shortened_before_action = Run_Simulations_Help.execute_single_modification_operator_decision_RL2(current_solution, 0, 0, 0, self.read_RL_data_day,timeslots_for_state_load_percentages_costs)


        #Specify observation space
        self.observation_space = (np.concatenate((percentage_array_loads_per_timeslot_highest_prices_shortened,percentage_array_loads_per_timeslot_lowest_prices_shortened))).reshape(-1)


        info = {}

        #Update auxilliary variables
        self.help_state_resoluting_solution_dicitionary = 0
        self.read_data_for_new_iteration = False

        return  self.observation_space, info


    def render (self):

        pass

    def step(self, action ):

        current_solution = self.list_of_solution_current_training_day[self.solution_of_current_file]

        #Change current solution if desired
        if use_resulting_state_after_action_as_current_solution == True and self.help_state_resoluting_solution_dicitionary !=0:
            current_solution = self.help_state_resoluting_solution_dicitionary

        peak_load_current_solution = current_solution['simulationObjective_maximumLoad_kW_combined'][0]
        costs_current_solution = current_solution['simulationObjective_costs_Euro_combined'][0]
        thermal_discomfort_current_solution = current_solution['simulationObjective_thermalDiscomfort_combined'][0]


        print(f"current_iteration_overall: {self.help_counter_solution_total}")
        action_from_timeslot = action[0]
        action_to_timeslot = action[1]
        action_shifting_percentage = minimum_shifting_percentage + ((maximum_shifting_percentage - minimum_shifting_percentage)/ number_of_discrete_shifting_actions) * action[2]


        # Execute the action in the external simulation and return the next observation, reward, done, and info
        result_costs, result_peak, result_DC, results_dict, percentage_array_loads_per_timeslot_highest_prices_shortened, percentage_array_loads_per_timeslot_lowest_prices_shortened, percentage_array_loads_per_timeslot_highest_prices_shortened_before_action, percentage_array_loads_per_timeslot_lowest_prices_shortened_before_action = Run_Simulations_Help.execute_single_modification_operator_decision_RL2(current_solution, action_from_timeslot, action_to_timeslot, action_shifting_percentage, self.read_RL_data_day,timeslots_for_state_load_percentages_costs)


        #calculate state
        observation_space = (np.concatenate((percentage_array_loads_per_timeslot_highest_prices_shortened, percentage_array_loads_per_timeslot_lowest_prices_shortened))).reshape(-1)


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


        # Adjust the reward
        if improvement_costs > 0:
            improvement_costs = improvement_costs * 2
            if improvement_peak >= 0 and result_DC[0] < Run_Simulations_Help.threshold_discomfort_local_search:
                improvement_costs = improvement_costs * 5

        reward = improvement_peak + improvement_costs + improvement_discomfort


        #Calculate statistics for the different actions





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
            improvement_discomfort = improvement_discomfort * 5
            if result_DC[0] >Run_Simulations_Help.threshold_discomfort_local_search + 0.2:
                done =True
        if  self.help_counter_iteration_current_day > number_of_iterations_per_day:
            #Change the read base solution by choosing the base solution from a new day
            if choose_days_randomly == True:
                random_index = random.randint(0, len(days_for_testing) - 1)
                chosen_day = days_for_testing[random_index]
            else:
                self.help_index_current_day+=1
                try:
                    chosen_day = days_for_testing[self.help_index_current_day]
                except:
                    self.help_index_current_day = 0
                    chosen_day = days_for_testing[self.help_index_current_day]
                    done = True
            self.read_RL_data_day = chosen_day
            self.solution_of_current_file = 0
            self.read_RL_data_iteration = 0
            self.help_counter_current_number_new_solution_per_solution = 0
            self.help_counter_iteration_current_day = 0
            print(f"-------New Day {chosen_day}------------")


        #Print auxiliary variables (for testing)
        print("")
        print(f"self.current_number_total_solutions_of_the_run: {self.current_number_total_solutions_of_the_run}")
        print(f"self.current_iteration_of_the_training_day: {self.current_iteration_of_the_training_day}")
        print(f"self.current_training_day: {self.current_training_day}")
        print(f"self.solution_of_current_file: {self.solution_of_current_file}")
        print(f"self.read_data_for_new_iteration: {self.read_data_for_new_iteration}")
        print( f"self.help_counter_current_number_new_solution_per_solution: {self.help_counter_current_number_new_solution_per_solution}")
        print(f"self.help_counter_solution_total: {self.help_counter_solution_total}")
        print(f"self.help_counter_iteration_current_day: {self.help_counter_iteration_current_day}")
        print(f"self.help_index_current_day: {self.help_index_current_day}")
        print(f"self.read_RL_data_day: {self.read_RL_data_day}")
        print(f"self.read_RL_data_iteration: {self.read_RL_data_iteration}")

        info = {}
        print("---------")
        print("")
        print("")
        return observation_space, reward, done, False, info


#Use Stable Baselines 3 to apply a RL algorithm on the environmetn
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
import stable_baselines3 as sb3




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
models_dir = r"C:\Users\wi9632\Desktop\Ergebnisse\DSM\RL\RL_Models\\" + string_run_name + "_PPO_" + random_string
logdir = r"C:\Users\wi9632\Desktop\Ergebnisse\DSM\RL\RL_Logs\\" + string_run_name + "_PPO_" + random_string
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

#Define the model directory (PPO, A2C, TD3, DQN)
model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.4)

#train and save the model
model.learn(total_timesteps=number_of_runs_for_the_algorithm - 2)
model.save(os.path.join(models_dir, 'trained_PPO_model'))


# Calculate the elapsed time in seconds
end_time = time.time()
elapsed_time = end_time - start_time

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print("")
print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")


