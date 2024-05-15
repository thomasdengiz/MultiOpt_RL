import os

### control
RUN_SIMULATIONS_MERGED_WITH_RL = True # run w/ or w/o RL; class derived from Run_Simulations and Run_Simulations_WithRL3 to reduce duplicate code
# further optimization parameters can be adjusted in Run_Simulations_Help
### directories
DIR_DATA = 'data/' # former data # TODO DSM folder in related files removen

# input data
DIR_INPUT_DATA = os.path.join(DIR_DATA, 'Input_Data/')
DIR_PRICE_ONE_MINUTE_DAYS = os.path.join(DIR_INPUT_DATA, 'Price_1Minute_Days/')
DIR_OUTSIDE_TEMPERATURE_ONE_MINUTE_DAYS = os.path.join(DIR_INPUT_DATA, 'Outside_Temperature_1Minute_Days/')
##
DIR_TRAINING_DATA = os.path.join(DIR_INPUT_DATA, 'Training_Data/')
DIR_RESULT = os.path.join(DIR_DATA, "Results/") # data results instead of results
DIR_MULTI_OPT_TEST = os.path.join(DIR_RESULT, "MultiOptTest/")
##

# instance bases
DIR_INSTANCE_BASE = os.path.join(DIR_RESULT, 'Instance Base/') # not used everywhere
DIR_CENTRAL_OPT_INSTANCE_BASE = os.path.join(DIR_RESULT, 'Centralized Optimization/Instance Base/') # not used everywhere
DIR_INSTANCE_PYMOO = os.path.join(DIR_RESULT, "Instance_Pymoo/")



# log files
DIR_LOGS = os.path.join(DIR_RESULT, "logs/") # to create
DIR_LOG_FILE = os.path.join(DIR_LOGS, 'log_results_Building_combined.txt')
#ERGEBNISSE_DIR = "/Ergebnisse/"


#RL
#DIR_RL = os.path.join(DIR_RESULT, "RL/")
# DIR_RL_LOGS = os.path.join(DIR_RESULT, "RL/logs")
# DIR_RL_INPUT = os.path.join(DIR_RESULT, "RL/inputs")
# DIR_RL_MODELS = os.path.join(DIR_RESULT, "RL/models")
DIR_RL = os.path.join(DIR_DATA, "Reinforcement_Learning/")
DIR_RL_LOGS = os.path.join(DIR_RL, "RL_logs/")
DIR_RL_INPUT = os.path.join(DIR_RL, "RL_Input/")
DIR_RL_MODELS = os.path.join(DIR_RL, "RL_Trained_Models/")