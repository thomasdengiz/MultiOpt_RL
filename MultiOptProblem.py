import numpy as np
from pymoo.core.problem import ElementwiseProblem
import ICSimulation
import Run_Simulations
import time
import SetUpScenarios

#Plot pareto front
import os
from datetime import datetime
currentDatetimeString = datetime.today().strftime('%d_%m_%Y_Time_%H_%M_%S')
simulationName = "Test"
folderName_WholeSimulation = currentDatetimeString + "_" + simulationName + "_BTCombined_" + str(SetUpScenarios.numberOfBuildings_Total)
folderPath_pymoo = "C:/Users/wi9632/Desktop/Ergebnisse/DSM/Instance_Pymoo/" + folderName_WholeSimulation
try:
    os.makedirs(folderPath_pymoo)
except OSError:
    print("Creation of the directory %s failed" % folderPath_pymoo)
else:
    print("Successfully created the directory %s" % folderPath_pymoo)

#Objectives and scenarios (only for output in this case)
optParameters = {
    'optimizationGoal_minimizePeakLoad': True,
    'optimizationGoal_minimizeCosts': True,
    'optimizationGoal_minimizeGas': False,
    'optimizationGoal_minimizeThermalDiscomfort': False,
    'optimizationGoal_minimizeSurplusEnergy': False,
    'optimization_1Objective': False,
    'optimization_2Objective': True,
    'optimization_3Objectives': False,
    'optimization_4Objectives': False,
    'objective_minimizePeakLoad_normalizationValue': 1,
    'objective_minimizeCosts_normalizationValue': 1,
    'objective_minimizeThermalDiscomfort_normalizationValue': 1,
    'objective_minimizeSurplusEnergy_normalizationValue': 1,
    'objective_minimizePeakLoad_weight': 0.5,
    'objective_minimizeCosts_weight': 0.5,
    'objective_minimizeThermalDiscomfort_weight': 0.5,
    'objective_minimizeSurplusEnergy_weight': 0.5,
    'epsilon_objective_minimizeCosts_Active': False,
    'epsilon_objective_minimizePeakLoad_Active': False,
    'epsilon_objective_minimizeGasConsumption_Active': False ,
    'epsilon_objective_minimizeThermalDiscomfort_Active':False,
    'epsilon_objective_minimizeCosts_TargetValue': 9999999,
    'epsilon_objective_minimizeMaximumLoad_TargetValue': 9999999,
    'epsilon_objective_minimizeGasConsumption_TargetValue': 9999999,
    'epsilon_objective_minimizeThermalDiscomfort_TargetValue': 9999999
}

currentDay = 200
execution_time = "00:06:00"
numberOfVariables = SetUpScenarios.numberOfTimeSlotsPerDay * (SetUpScenarios.numberOfBuildings_BT1 * 3 + SetUpScenarios.numberOfBuildings_BT2 * 2 + SetUpScenarios.numberOfBuildings_BT3 * 1 +  SetUpScenarios.numberOfBuildings_BT4 * 1 +  SetUpScenarios.numberOfBuildings_BT5 * 2 +  SetUpScenarios.numberOfBuildings_BT6 * 3 +   SetUpScenarios.numberOfBuildings_BT7 * 2)

# Measure wall-clock time and CPU time
start_time = time.time()
start_cpu = time.process_time()

def transfer_simulation_input_into_decisionvariable (outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5,outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7):

    #Assigne the values of the inptut vectors for the simulation (called outputVectors here; htey will be used as input for the simulation and evaluation) to the decicion variable x
    currentPositionInVariableX = 0
    x = np.zeros (numberOfVariables)

    for a in range (0, SetUpScenarios.numberOfBuildings_BT1 ):
        for b in range (0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x [currentPositionInVariableX] = outputVector_heatGenerationCoefficientSpaceHeating_BT1[a][b]
            currentPositionInVariableX += 1
    for a in range(0, SetUpScenarios.numberOfBuildings_BT1):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_heatGenerationCoefficientDHW_BT1[a][b]
            currentPositionInVariableX += 1
    for a in range(0, SetUpScenarios.numberOfBuildings_BT1):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_chargingPowerEV_BT1[a][b] / SetUpScenarios.chargingPowerMaximal_EV
            currentPositionInVariableX += 1


    for a in range(0, SetUpScenarios.numberOfBuildings_BT2):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] =  outputVector_heatGenerationCoefficientSpaceHeating_BT2[a][b]
            currentPositionInVariableX += 1
    for a in range(0, SetUpScenarios.numberOfBuildings_BT2):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_heatGenerationCoefficientDHW_BT2[a][b]
            currentPositionInVariableX += 1

    for a in range(0, SetUpScenarios.numberOfBuildings_BT3):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_chargingPowerEV_BT3[a][b] / SetUpScenarios.chargingPowerMaximal_EV
            currentPositionInVariableX += 1

    for a in range(0, SetUpScenarios.numberOfBuildings_BT4):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_heatGenerationCoefficientSpaceHeating_BT4[a][b]
            currentPositionInVariableX += 1

    for a in range(0, SetUpScenarios.numberOfBuildings_BT5):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_chargingPowerBAT_BT5[a][b] / SetUpScenarios.chargingPowerMaximal_BAT
            currentPositionInVariableX += 1
    for a in range(0, SetUpScenarios.numberOfBuildings_BT5):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_disChargingPowerBAT_BT5[a][b] / SetUpScenarios.chargingPowerMaximal_BAT
            currentPositionInVariableX += 1


    for a in range (0, SetUpScenarios.numberOfBuildings_BT6 ):
        for b in range (0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x [currentPositionInVariableX] = outputVector_heatGenerationCoefficient_GasBoiler_BT6[a][b]
            currentPositionInVariableX += 1
    for a in range(0, SetUpScenarios.numberOfBuildings_BT6):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6[a][b]
            currentPositionInVariableX += 1
    for a in range(0, SetUpScenarios.numberOfBuildings_BT6):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_heatTransferCoefficient_StorageToRoom_BT6[a]/  SetUpScenarios.maximalPowerHeatingSystem
            currentPositionInVariableX += 1


    for a in range (0, SetUpScenarios.numberOfBuildings_BT7 ):
        for b in range (0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] =  outputVector_heatGenerationCoefficient_GasBoiler_BT7[a][b]
            currentPositionInVariableX += 1
    for a in range(0, SetUpScenarios.numberOfBuildings_BT7):
        for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
            x[currentPositionInVariableX] = outputVector_electricalPowerFanHeater_BT7[a][b] / SetUpScenarios.electricalPowerFanHeater_Stage3
            currentPositionInVariableX += 1

    return  x


def transfer_decisionvariables_into_simulation_inputs (x):
        #Assigne the values of the generated variable x (by Pymoo) to the inptut vectors for the simulation (called outputVectors here; htey will be used as input for the simulation and evaluation)
        currentPositionInVariableX = 0
        outputVector_heatGenerationCoefficientSpaceHeating_BT1 = np.zeros((SetUpScenarios.numberOfBuildings_BT1, SetUpScenarios.numberOfTimeSlotsPerDay))
        outputVector_heatGenerationCoefficientDHW_BT1 = np.zeros((SetUpScenarios.numberOfBuildings_BT1, SetUpScenarios.numberOfTimeSlotsPerDay))
        outputVector_chargingPowerEV_BT1 = np.zeros((SetUpScenarios.numberOfBuildings_BT1, SetUpScenarios.numberOfTimeSlotsPerDay))
        for a in range (0, SetUpScenarios.numberOfBuildings_BT1 ):
            for b in range (0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_heatGenerationCoefficientSpaceHeating_BT1[a][b] = x [currentPositionInVariableX]
                currentPositionInVariableX += 1
        for a in range(0, SetUpScenarios.numberOfBuildings_BT1):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_heatGenerationCoefficientDHW_BT1[a][b]  = x[currentPositionInVariableX]
                currentPositionInVariableX += 1
        for a in range(0, SetUpScenarios.numberOfBuildings_BT1):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_chargingPowerEV_BT1[a][b]  = x[currentPositionInVariableX] * SetUpScenarios.chargingPowerMaximal_EV
                currentPositionInVariableX += 1


        outputVector_heatGenerationCoefficientSpaceHeating_BT2 = np.zeros((SetUpScenarios.numberOfBuildings_BT2, SetUpScenarios.numberOfTimeSlotsPerDay))
        outputVector_heatGenerationCoefficientDHW_BT2 = np.zeros((SetUpScenarios.numberOfBuildings_BT2, SetUpScenarios.numberOfTimeSlotsPerDay))
        for a in range(0, SetUpScenarios.numberOfBuildings_BT2):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_heatGenerationCoefficientSpaceHeating_BT2[a][b] = x[currentPositionInVariableX]
                currentPositionInVariableX += 1
        for a in range(0, SetUpScenarios.numberOfBuildings_BT2):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_heatGenerationCoefficientDHW_BT2[a][b]  = x[currentPositionInVariableX]
                currentPositionInVariableX += 1

        outputVector_chargingPowerEV_BT3 = np.zeros((SetUpScenarios.numberOfBuildings_BT3, SetUpScenarios.numberOfTimeSlotsPerDay))
        for a in range(0, SetUpScenarios.numberOfBuildings_BT3):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_chargingPowerEV_BT3[a][b] = x[currentPositionInVariableX] * SetUpScenarios.chargingPowerMaximal_EV
                currentPositionInVariableX += 1

        outputVector_heatGenerationCoefficientSpaceHeating_BT4 = np.zeros((SetUpScenarios.numberOfBuildings_BT4, SetUpScenarios.numberOfTimeSlotsPerDay))
        for a in range(0, SetUpScenarios.numberOfBuildings_BT4):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_heatGenerationCoefficientSpaceHeating_BT4[a][b]  = x[currentPositionInVariableX]
                currentPositionInVariableX += 1

        outputVector_chargingPowerBAT_BT5 = np.zeros((SetUpScenarios.numberOfBuildings_BT5, SetUpScenarios.numberOfTimeSlotsPerDay))
        outputVector_disChargingPowerBAT_BT5 = np.zeros((SetUpScenarios.numberOfBuildings_BT5, SetUpScenarios.numberOfTimeSlotsPerDay))
        for a in range(0, SetUpScenarios.numberOfBuildings_BT5):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_chargingPowerBAT_BT5[a][b]  = x[currentPositionInVariableX] * SetUpScenarios.chargingPowerMaximal_BAT
                currentPositionInVariableX += 1
        for a in range(0, SetUpScenarios.numberOfBuildings_BT5):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_disChargingPowerBAT_BT5[a][b]  = x[currentPositionInVariableX] * SetUpScenarios.chargingPowerMaximal_BAT
                currentPositionInVariableX += 1

        outputVector_heatGenerationCoefficient_GasBoiler_BT6 = np.zeros((SetUpScenarios.numberOfBuildings_BT6, SetUpScenarios.numberOfTimeSlotsPerDay))
        outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6 = np.zeros((SetUpScenarios.numberOfBuildings_BT6, SetUpScenarios.numberOfTimeSlotsPerDay))
        outputVector_heatTransferCoefficient_StorageToRoom_BT6 = np.zeros((SetUpScenarios.numberOfBuildings_BT6, SetUpScenarios.numberOfTimeSlotsPerDay))
        for a in range (0, SetUpScenarios.numberOfBuildings_BT6 ):
            for b in range (0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_heatGenerationCoefficient_GasBoiler_BT6[a][b]  = x [currentPositionInVariableX]
                currentPositionInVariableX += 1
        for a in range(0, SetUpScenarios.numberOfBuildings_BT6):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6[a][b]  = x[currentPositionInVariableX]
                currentPositionInVariableX += 1
        for a in range(0, SetUpScenarios.numberOfBuildings_BT6):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_heatTransferCoefficient_StorageToRoom_BT6[a] = x[currentPositionInVariableX] * SetUpScenarios.maximalPowerHeatingSystem
                currentPositionInVariableX += 1

        outputVector_heatGenerationCoefficient_GasBoiler_BT7 = np.zeros((SetUpScenarios.numberOfBuildings_BT7, SetUpScenarios.numberOfTimeSlotsPerDay))
        outputVector_electricalPowerFanHeater_BT7 = np.zeros((SetUpScenarios.numberOfBuildings_BT7, SetUpScenarios.numberOfTimeSlotsPerDay))
        for a in range (0, SetUpScenarios.numberOfBuildings_BT7 ):
            for b in range (0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_heatGenerationCoefficient_GasBoiler_BT7[a][b]  = x [currentPositionInVariableX]
                currentPositionInVariableX += 1
        for a in range(0, SetUpScenarios.numberOfBuildings_BT7):
            for b in range(0, SetUpScenarios.numberOfTimeSlotsPerDay):
                outputVector_electricalPowerFanHeater_BT7[a][b]  = x[currentPositionInVariableX] * SetUpScenarios.electricalPowerFanHeater_Stage3
                currentPositionInVariableX += 1

        return  outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5,outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7

evaluation_counter = 0
class MyProblem(ElementwiseProblem):

    def __init__(self, pathForCreatingResults, secondObjectiveComfort):
        arrayLowerBounds = np.empty(numberOfVariables)
        arrayLowerBounds.fill(0)
        arrayUpperBounds = np.empty(numberOfVariables)
        arrayUpperBounds.fill(1)
        super().__init__(n_var=numberOfVariables ,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=arrayLowerBounds,
                         xu=arrayUpperBounds)

    def _evaluate(self, x, out, *args, **kwargs):
        import ICSimulation
        global evaluation_counter
        evaluation_counter = evaluation_counter + 1
        #print(f"evaluation_counter: {evaluation_counter}")


        outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7 = transfer_decisionvariables_into_simulation_inputs(x)

        preCorrectSchedules_AvoidingFrequentStarts = False
        use_local_search = True
        simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, simulationResult_electricalLoad_combined , price_array, simulationInput_BT1_availabilityPattern, combined_array_thermal_discomfort, outputVector_BT1_heatGenerationCoefficientSpaceHeating_corrected, outputVector_BT1_heatGenerationCoefficientDHW_corrected, outputVector_BT1_chargingPowerEV_corrected, outputVector_BT2_heatGenerationCoefficientSpaceHeating_corrected, outputVector_BT2_heatGenerationCoefficientDHW_corrected, outputVector_BT3_chargingPowerEV_corrected, outputVector_BT4_heatGenerationCoefficientSpaceHeating_corrected, outputVector_BT5_chargingPowerBAT_corrected, outputVector_BT5_disChargingPowerBAT_corrected, outputVector_BT6_heatGenerationCoefficient_GasBoiler, outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement, outputVector_BT6_heatTransferCoefficient_StorageToRoom, outputVector_BT7_heatGenerationCoefficient_GasBoiler, outputVector_BT7_electricalPowerFanHeater, combined_array_thermal_discomfort, thermal_discomfort_space_heating_BT1, thermal_discomfort_dhw_BT1, thermal_discomfort_space_heating_BT2, thermal_discomfort_dhw_BT2, thermal_discomfort_space_heating_BT4 = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5,outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, pathForCreatingResults, preCorrectSchedules_AvoidingFrequentStarts, optParameters, use_local_search)





        firstObjective = simulationObjective_costs_Euro_combined[0]
        if secondObjectiveComfort == True:
            secondObjective = simulationObjective_thermalDiscomfort_combined [0]
        else:
            secondObjective = simulationObjective_maximumLoad_kW_combined[0]

        constraint_violation_thermal_discomfort = 0
        if simulationObjective_thermalDiscomfort_combined[0] > Run_Simulations.threshold_discomfort_local_search:
            constraint_violation_thermal_discomfort = simulationObjective_thermalDiscomfort_combined[0] - Run_Simulations.threshold_discomfort_local_search

        g1 = constraint_violation_thermal_discomfort

        out["F"] = [firstObjective, secondObjective]
        out["G"] = [g1]




pathForCreatingResults =  "C:/Users/wi9632/Desktop/Ergebnisse/DSM/MultiOptTest"
secondObjectiveComfort = False
problem = MyProblem(pathForCreatingResults, secondObjectiveComfort)


#Definition of the algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling


#Create feasible initial solution by using the conventional contorl

# run simulation environment with conventional control
use_local_search = True
indexOfBuildingsOverall_BT1 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT1 + 1)]
indexOfBuildingsOverall_BT2 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT2 + 1)]
indexOfBuildingsOverall_BT3 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT3 + 1)]
indexOfBuildingsOverall_BT4 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT4 + 1)]
indexOfBuildingsOverall_BT5 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT5 + 1)]
indexOfBuildingsOverall_BT6 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT6 + 1)]
indexOfBuildingsOverall_BT7 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT7 + 1)]


simulationObjective_surplusEnergy_kWh_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, simulationResult_electricalLoad_combined, price_array, simulationInput_BT1_availabilityPattern, combined_array_thermal_discomfort, outputVector_BT1_heatGenerationCoefficientSpaceHeating, outputVector_BT1_heatGenerationCoefficientDHW, outputVector_BT1_chargingPowerEV, outputVector_BT2_heatGenerationCoefficientSpaceHeating, outputVector_BT2_heatGenerationCoefficientDHW, outputVector_BT3_chargingPowerEV, outputVector_BT4_heatGenerationCoefficientSpaceHeating, outputVector_BT5_chargingPowerBAT, outputVector_BT5_disChargingPowerBAT, outputVector_BT6_heatGenerationCoefficient_GasBoiler, outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement, outputVector_BT6_heatTransferCoefficient_StorageToRoom, outputVector_BT7_heatGenerationCoefficient_GasBoiler, outputVector_BT7_electricalPowerFanHeater, combined_array_thermal_discomfort, simulationResult_thermalDiscomfort_BT1, simulationResult_thermalDiscomfort_BT2, simulationResult_thermalDiscomfort_BT3, simulationResult_thermalDiscomfort_BT4, simulationResult_thermalDiscomfort_BT5, simulationResult_thermalDiscomfort_BT6, simulationResult_thermalDiscomfort_BT7 = ICSimulation.simulateDays_ConventionalControl(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, pathForCreatingResults, use_local_search)
#Create initial solution vector
initial_solution_conventional_control = transfer_simulation_input_into_decisionvariable (outputVector_BT1_heatGenerationCoefficientSpaceHeating, outputVector_BT1_heatGenerationCoefficientDHW, outputVector_BT1_chargingPowerEV, outputVector_BT2_heatGenerationCoefficientSpaceHeating, outputVector_BT2_heatGenerationCoefficientDHW, outputVector_BT3_chargingPowerEV, outputVector_BT4_heatGenerationCoefficientSpaceHeating, outputVector_BT5_chargingPowerBAT, outputVector_BT5_disChargingPowerBAT, outputVector_BT6_heatGenerationCoefficient_GasBoiler, outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement, outputVector_BT6_heatTransferCoefficient_StorageToRoom, outputVector_BT7_heatGenerationCoefficient_GasBoiler, outputVector_BT7_electricalPowerFanHeater)



#Define class for hybrid sampling to include initial solution
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling


class HybridSampling(Sampling):

    def __init__(self, base_sampler=None, candidates=None):
        """Sampling class that uses candidate solutions

        Parameters
        ----------
        base_sampler : Sampling | None, optional
            Base pymoo sampler. If None, uses FloatRandomSampling, by default None

        candidates : numpy.array | None, optional
            Candidate solutions, by default None
        """
        super().__init__()
        if base_sampler is None:
            base_sampler = FloatRandomSampling()
        self.base_sampler = base_sampler

        self.candidates = candidates
        if self.N > 0:
            self._do = self._do_replace
        else:
            self._do = self.base_sampler._do

    @property
    def N(self):
        if self.candidates is None:
            return 0
        else:
            return np.shape(self.candidates)[0]

    def _do_replace(self, problem, n_samples, **kwargs):

        candidates = np.array(self.candidates)
        if np.shape(candidates)[1] != problem.n_var:
            raise ValueError("Candidates must have dimension equals n_var on axis=1")
        else:
            pass

        if n_samples == self.N:
            return np.array(self.candidates)
        else:
            X = self.base_sampler._do(problem, n_samples, **kwargs)

        if n_samples < self.N:
            reps = np.random.choice(self.N)
            X = np.array(self.candidates)[reps]
        else:
            X[:self.N, ...] = np.array(self.candidates)

        return X



algorithm = NSGA2(
    pop_size=100,
    n_offsprings=100,
    #sampling=FloatRandomSampling(),
    sampling=HybridSampling(candidates=[initial_solution_conventional_control]),
    crossover=SBX(prob=0.7, eta=20),
    mutation=PM(eta=40),
    eliminate_duplicates=True
)



#algorithm.setup(problem, x0=initial_solution_conventional_control)


from pymoo.termination import get_termination
#termination = get_termination("n_gen", 35)

termination = get_termination("time", execution_time)
from pymoo.optimize import minimize
res = minimize(problem,
               algorithm,
               termination,
               seed=None,
               save_history=True,
               verbose=True,
               n_jobs=8)

#Get the end result data (F=Objective space, X=decision space)
X = res.X
F = res.F

# Create the new DataFrame by dropping duplicate rows based on rounded values
rounded_F = np.round(F, decimals=2)
seen_values = set()
non_duplicate_values = []

for i, row in enumerate(rounded_F):
    value_tuple = tuple(row)

    if value_tuple not in seen_values:
        seen_values.add(value_tuple)
        non_duplicate_values.append(i)

non_duplicate_values = np.array(non_duplicate_values)
result = F[non_duplicate_values]

#Print the profiles after the last iteration of the optimization algorithm
import ICSimulation
for i in range (0,len(X)):
    if ~np.in1d(i, non_duplicate_values):
        continue
    indexOfBuildingsOverall_BT1 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT1 + 1)]
    indexOfBuildingsOverall_BT2 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT2 + 1)]
    indexOfBuildingsOverall_BT3 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT3 + 1)]
    indexOfBuildingsOverall_BT4 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT4 + 1 )]
    indexOfBuildingsOverall_BT5 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT5 + 1 )]
    indexOfBuildingsOverall_BT6 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT6 + 1 )]
    indexOfBuildingsOverall_BT7 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT7 + 1 )]

    arrayOneSOlution = X[i]
    outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7 = transfer_decisionvariables_into_simulation_inputs(X[i])

    preCorrectSchedules_AvoidingFrequentStarts = False

    use_local_search = False


    if i == 0:
        # create "Individual Solutions" folder
        folder_path_individual_solutions = os.path.join(folderPath_pymoo, "Individual Solutions")
        try:
            os.makedirs(folder_path_individual_solutions)
        except OSError:
            print(f"Creation of the directory {folder_path_individual_solutions} failed")

    # create "Solution_" subfolders inside "Individual Solutions" folder
    subfolder_name = "Solution_" + str(i+1)
    folder_path_single_result = os.path.join(folder_path_individual_solutions, subfolder_name)
    try:
        os.makedirs(folder_path_single_result)
    except OSError:
        print(f"Creation of the directory {folder_path_single_result} failed")

    simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined,simulationObjective_thermalDiscomfort_combined,simulationObjective_gasConsumptionkWh_combined,  simulationObjective_combinedScore_combined = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5,outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, folder_path_single_result, preCorrectSchedules_AvoidingFrequentStarts, optParameters, use_local_search)


import pandas as pd
df_results = pd.DataFrame(F, columns=["Costs", "Peak Load"])





# Print results to csv
titleOfThePlot = "NSGA2 - Day: " + str(currentDay) + " - "
appendixResultFile = "NSGA2" + "_Day" + str(currentDay) + ""
if SetUpScenarios.numberOfBuildings_BT1 > 0:
    titleOfThePlot = titleOfThePlot + "BT1: " + str(SetUpScenarios.numberOfBuildings_BT1) + ", "
    appendixResultFile = appendixResultFile + "_BT1_" + str(SetUpScenarios.numberOfBuildings_BT1)
if SetUpScenarios.numberOfBuildings_BT2 > 0:
    titleOfThePlot = titleOfThePlot + "BT2: " + str(SetUpScenarios.numberOfBuildings_BT2) + ", "
    appendixResultFile = appendixResultFile + "_BT2_" + str(SetUpScenarios.numberOfBuildings_BT2)
if SetUpScenarios.numberOfBuildings_BT3 > 0:
    titleOfThePlot = titleOfThePlot + "BT3: " + str(SetUpScenarios.numberOfBuildings_BT3) + ", "
    appendixResultFile = appendixResultFile + "_BT3_" + str(SetUpScenarios.numberOfBuildings_BT3)
if SetUpScenarios.numberOfBuildings_BT4 > 0:
    titleOfThePlot = titleOfThePlot + "BT4: " + str(SetUpScenarios.numberOfBuildings_BT4) + ", "
    appendixResultFile = appendixResultFile + "_BT4_" + str(SetUpScenarios.numberOfBuildings_BT4)
if SetUpScenarios.numberOfBuildings_BT5 > 0:
    titleOfThePlot = titleOfThePlot + "BT5: " + str(SetUpScenarios.numberOfBuildings_BT5) + ", "
    appendixResultFile = appendixResultFile + "_BT5_" + str(SetUpScenarios.numberOfBuildings_BT5)
if SetUpScenarios.numberOfBuildings_BT6 > 0:
    titleOfThePlot = titleOfThePlot + "BT6: " + str(SetUpScenarios.numberOfBuildings_BT6) + ", "
    appendixResultFile = appendixResultFile + "_BT6_" + str(SetUpScenarios.numberOfBuildings_BT6)
if SetUpScenarios.numberOfBuildings_BT7 > 0:
    titleOfThePlot = titleOfThePlot + "BT7: " + str(SetUpScenarios.numberOfBuildings_BT7) + ", "
    appendixResultFile = appendixResultFile + "_BT7_" + str(SetUpScenarios.numberOfBuildings_BT7)
titleOfThePlot = titleOfThePlot[:-2]
df_results['Costs'] = df_results['Costs'].round(2)
df_results['Peak Load'] = df_results['Peak Load'].round(2)


df_results.to_csv(folderPath_pymoo + "Peak_Costs" + appendixResultFile + ".csv",index=False, sep=";")

# Create figure with pareto front
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
font_size_title_Pareto_Plot  = 14

matplotlib.use('Agg')

if df_results.empty:
    print("Error: df_results DataFrame is empty.")
else:
    # Create an empty DataFrame to store the Pareto-efficient solutions
    pareto_front = pd.DataFrame(columns=['Costs', 'Peak Load'])

    # Loop through all solutions in the DataFrame
    for i, row in df_results.iterrows():
        # Assume the current solution is Pareto-efficient until proven otherwise
        is_efficient = True
        # Loop through all existing solutions in the Pareto front
        for j, pareto_row in pareto_front.iterrows():
            # Check if the current solution is dominated by any existing solution in the Pareto front
            if (row['Costs'] >= pareto_row['Costs'] and row['Peak Load'] >= pareto_row[
                'Peak Load']):
                is_efficient = False
                break
            # Check if the current solution dominates any existing solution in the Pareto front
            elif (row['Costs'] <= pareto_row['Costs'] and row['Peak Load'] <= pareto_row[
                'Peak Load']):
                pareto_front = pareto_front.drop(j)
        # If the current solution is Pareto-efficient, add it to the Pareto front
        if is_efficient:
            pareto_front = pareto_front = pd.concat([pareto_front, row.to_frame().T], ignore_index=True)

    if pareto_front.empty:
        print("Error: pareto_front DataFrame is empty.")
    else:
        # Sort the Pareto front by 'Costs' and 'Peak Load'
        pareto_front = pareto_front.sort_values(['Costs', 'Peak Load'], ascending=[True, True])
        # Reset the index of the Pareto front DataFrame
        pareto_front = pareto_front.reset_index(drop=True)

        # Plot Pareto efficient solutions with line
        plt.scatter(pareto_front['Costs'], pareto_front['Peak Load'], color='blue')
        plt.plot(pareto_front['Costs'], pareto_front['Peak Load'], color='red')
        plt.xlabel('Costs', fontsize=15)
        plt.ylabel('Peak Load', fontsize=15)
        if font_size_title_Pareto_Plot > 0:
            plt.title(titleOfThePlot, fontsize=font_size_title_Pareto_Plot)
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.subplots_adjust(left=0.15)
        plt.savefig(folderPath_pymoo + '/PFront_Line_' + appendixResultFile + '.png', dpi=100)
        # Clear current figure
        plt.clf()
        # Plot Pareto efficient solutions without line
        plt.scatter(pareto_front['Costs'], pareto_front['Peak Load'], color='blue')
        plt.xlabel('Costs', fontsize=14)
        plt.ylabel('Peak Load', fontsize=14)
        plt.subplots_adjust(left=0.15)
        if font_size_title_Pareto_Plot > 0:
            plt.title(titleOfThePlot, fontsize=font_size_title_Pareto_Plot)
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.savefig(folderPath_pymoo + '/PFront_' + appendixResultFile + '.png',dpi=100)

end_time = time.time()
execution_time = end_time - start_time
end_cpu = time.process_time()
execution_cpu = end_cpu - start_cpu
print("")
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)
formatted_time = "{:02d} hours, {:02d} minutes, {:02d} seconds".format(int(hours), int(minutes), int(seconds))
print("Normal Execution Time:", formatted_time)

hours, remainder = divmod(execution_cpu, 3600)
minutes, seconds = divmod(remainder, 60)
formatted_time = "{:02d} hours, {:02d} minutes, {:02d} seconds".format(int(hours), int(minutes), int(seconds))
print("CPU  Time:", formatted_time)
