# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:26:50 2021

@author: wi9632
"""
import SetUpScenarios 
import numpy as np
import os
from datetime import datetime


#Set up

# Specify the used optimization methods
useCentralizedOptimization = False
useDecentralizedOptimization = False
useSupervisedLearning = True
useReinforcementLearning = False
useConventionalControl = False
generateTrainingData = False

#Objectives and scenarios

optimizationGoal_minimizeSurplusEnergy = False
optimizationGoal_minimizePeakLoad = False
optimizationGoal_minimizeCosts = True

useCorrectionsAtTheEndOfDay = False


# Choose internal Controller 
run_simulateDays_NoAddtionalController_ANNSchedule = False
run_simulateDays_WithAddtionalController_Schedule =True
useInternalControllerToOverruleActions_simulateDays_WithAddtionalController_Schedule = True

useInternalControllerForRL = True



#Maximal starting times for the heating devices
considerMaximumNumberOfStartsHP_Combined = True
considerMaxiumNumberOfStartsHP_Individual = False
considerMaxiumNumberOfStartsHP_MFH_Individual = True
isHPAlwaysSwitchedOn = False
maximumNumberOfStarts_Combined = 4
maximumNumberOfStarts_Individual = 2

#Parameters for the internal controller
minimalRunTimeHeatPump = 0
timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat =0

#Adjust parameters to the time resolution
if SetUpScenarios.timeResolution_InMinutes ==1:
    minimalRunTimeHeatPump = 30
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 5
if SetUpScenarios.timeResolution_InMinutes ==5:
    minimalRunTimeHeatPump = 6
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 4
if SetUpScenarios.timeResolution_InMinutes ==10:
    minimalRunTimeHeatPump = 3
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 4
if SetUpScenarios.timeResolution_InMinutes ==15:
    minimalRunTimeHeatPump = 2
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 2
if SetUpScenarios.timeResolution_InMinutes ==30:
    minimalRunTimeHeatPump = 1
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 1
if SetUpScenarios.timeResolution_InMinutes ==60:
    minimalRunTimeHeatPump = 1
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 1
additionalNumberOfAllowedStarts = 7
additionalNumberOfAllowedStarts_BeforeConsideringMinimalRuntime = 3

if useCorrectionsAtTheEndOfDay ==False:
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = -1

minimalModulationDegreeOfTheMaximumPowerInCaseOfANecessaryCorrection = 0.7 # Should be between 0.5 and 1. Only quantifies the minimal degree. If no new peak is created by this action in the simulation , the value will be higher


differntWeigthsForTheOptimization_2Objectives = [(1.0, 0.0), (0.0, 1.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75)]
differntWeigthsForTheOptimization_3Objectives = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.33, 0.33, 0.33), 
                                                 (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5)]


#Values are determined below
optimization_1Objective = False
optimization_2Objective = False
optimization_3Objectives = False

#Determine the boolean values of the variables for the number of objectives considered
if ((optimizationGoal_minimizeSurplusEnergy == True and optimizationGoal_minimizePeakLoad == False and optimizationGoal_minimizeCosts == False) or
    (optimizationGoal_minimizeSurplusEnergy == False and optimizationGoal_minimizePeakLoad == True and optimizationGoal_minimizeCosts == False) or 
    (optimizationGoal_minimizeSurplusEnergy == False and optimizationGoal_minimizePeakLoad == False and optimizationGoal_minimizeCosts == True)):
    optimization_1Objective = True
    print("1 Objectives")
if ((optimizationGoal_minimizeSurplusEnergy == True and optimizationGoal_minimizePeakLoad == True and optimizationGoal_minimizeCosts == False) or
    (optimizationGoal_minimizeSurplusEnergy == False and optimizationGoal_minimizePeakLoad == True and optimizationGoal_minimizeCosts == True) or 
    (optimizationGoal_minimizeSurplusEnergy == True and optimizationGoal_minimizePeakLoad == False and optimizationGoal_minimizeCosts == True)):
    optimization_2Objective = True
    print("2 Objectives")
if (optimizationGoal_minimizeSurplusEnergy == True and optimizationGoal_minimizePeakLoad == True and optimizationGoal_minimizeCosts == True):
    optimization_3Objectives = True
    print("3 Objectives")


objective_minimizeSurplusEnergy_normalizationValue = 1
objective_minimizePeakLoad_normalizationValue = 1
objective_minimizeCosts_normalizationValue = 1

objective_minimizeSurplusEnergy_weight = 0.5
objective_minimizePeakLoad_weight = 0.5
objective_minimizeCosts_weight = 0.5


#Constants
ML_METHOD_RANDOM_FOREST = "Random_Forest"
ML_METHOD_MULTI_LAYER_PERCEPTRON = "Multi_Layer_Perceptron"
ML_METHOD_GRADIENT_BOOSTING = "Gradient_Boosting"
ML_METHOD_RNN = "RNN"
ML_METHOD_LSTM = "LSTM"

OPT_OBJECTIVE_MIN_PEAK = 'Min_Peak'
OPT_OBJECTIVE_MIN_SURPLUS = 'Min_SurplusEnergy'
OPT_OBJECTIVE_MIN_COSTS = 'Min_Costs'

numberOfBuildingDataOverall = 20

#TODO: Continue Add constants to the run methods, call RNN, LSTM

# Weights for evaluating the constraint violations
weight_total_ConstraintViolation_BufferStorageTemperatureRange_combined = 1
weight_total_ConstraintViolation_DHWTankRange_combined =1
weight_total_ConstraintViolation_BufferStorageTemperatureLastValue_combined = 1
weight_total_ConstraintViolation_DHWTankLastValue_combined = 1
weight_total_ConstraintViolation_SOCRangeOfTheEV_combined  = 1
weight_total_ConstraintViolation_SOCOfTheEVLastValue_combined  = 1  # suggestion 0.5
weight_total_ConstraintViolation_ChargingPowerOfTheEV_combined  = 1
weight_total_ConstraintViolation_numberOfStarts_Combined_combined = 1
weight_total_ConstraintViolation_SOCRangeOfTheBAT_combined  = 1
   
weight_total_ConstraintViolation_BufferStorageTemperatureRange_CorrectionLimit_Combined =1
weight_total_ConstraintViolation_DHWTankRange_CorrectionLimit_Combined = 1
weight_total_ConstraintViolation_BufferStorageTemperatureRange_PhysicalLimit_Combined = 1
weight_total_ConstraintViolation_DHWTankRange_PhysicalLimit_Combined  = 1
weight_total_ConstraintViolation_BufferStorageTemperatureLastValue_CorrectionLimit_Combined = 1
weight_total_ConstraintViolation_DHWTankLastValue_CorrectionLimit_Combined = 1
weight_total_ConstraintViolation_SOCOfTheEVLastValue_CorrectionLimit_Combined  = 1
weight_total_ConstraintViolation_numberOfStarts_CorrectionLimit_Combined_Combined  =1




daysOfTheYearForSimulation_Testing = [ 5]


###################################################################################################################################################################################################

def generateTrainingDataForML():
    print("Generate Training Data")
    import pandas as pd
    
    for buildingIndex in range (1, 21):
        daysOfTheYearForSimulation_Training_Period1 = [i for i in range (1, 91)]
        daysOfTheYearForSimulation_Training_Period2 = [i for i in range (274, 366)]
            

        pathForCreatingTheResultData_Centralized = folderPath_WholeSimulation + "/Centralized/Min_Surplus_Basic_PV_"+ str(SetUpScenarios.averagePVPeak) + "_kWp_" +  str(SetUpScenarios.timeResolution_InMinutes) + "_Min/BT5_HH" + str(buildingIndex) + "/"

        indexOfBuildingsOverall_BT1 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT1 + 1)]
        indexOfBuildingsOverall_BT2 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT2 + 1)]
        indexOfBuildingsOverall_BT3 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT3 + 1)]
        indexOfBuildingsOverall_BT4 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT4 + 1)]
        indexOfBuildingsOverall_BT5 = [buildingIndex]
        
        numberOfDaysWithNegativeScore = 0
        df_results_overall = pd.DataFrame(columns =['Building','Day', 'Surplus Energy [kWh]', 'Peak Load [kW]', 'Costs [Euro]', 'Score', 'Negative Score'])

        for currentDay in daysOfTheYearForSimulation_Training_Period1:
            print("buildingIndex for generating Data: ", buildingIndex)
            print("currentDay for generating Data: ", currentDay)
            print()
            print()
            outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT1, outputVectorOptimization_heatGenerationCoefficientDHW_BT1, outputVectorOptimization_chargingPowerEV_BT1, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT2, outputVectorOptimization_heatGenerationCoefficientDHW_BT2, outputVectorOptimization_chargingPowerEV_BT3, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT4,  outputVectorOptimization_chargingPowerBAT_BT5, outputVectorOptimization_disChargingPowerBAT_BT5 =   Building_Combined.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4,indexOfBuildingsOverall_BT5, currentDay)

            #Call the internal controller with the schedules
            overruleActions = False  
            simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, negativeScore_total_overall = ICSimulation.simulateDays_WithAddtionalController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, overruleActions, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT1, outputVectorOptimization_heatGenerationCoefficientDHW_BT1, outputVectorOptimization_chargingPowerEV_BT1, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT2, outputVectorOptimization_heatGenerationCoefficientDHW_BT2, outputVectorOptimization_chargingPowerEV_BT3, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT4, outputVectorOptimization_chargingPowerBAT_BT5, outputVectorOptimization_disChargingPowerBAT_BT5, pathForCreatingTheResultData_Centralized)
            if negativeScore_total_overall > 0.1:
                numberOfDaysWithNegativeScore = numberOfDaysWithNegativeScore + 1
                
            #Save results of the runs in a csv file 
            df_results_overall.loc[len(df_results_overall)] = [buildingIndex, currentDay, simulationObjective_surplusEnergy_kWh_combined[0], simulationObjective_maximumLoad_kW_combined[0], simulationObjective_costs_Euro_combined[0], simulationObjective_combinedScore_combined[0], negativeScore_total_overall[0]]
            df_results_overall.to_csv( pathForCreatingTheResultData_Centralized  + "/results_overall.csv", sep=";") 
            
        for currentDay in daysOfTheYearForSimulation_Training_Period2:
            print("buildingIndex for generating Data: ", buildingIndex)
            print("currentDay for generating Data: ", currentDay)
            print()
            print()
            outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT1, outputVectorOptimization_heatGenerationCoefficientDHW_BT1, outputVectorOptimization_chargingPowerEV_BT1, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT2, outputVectorOptimization_heatGenerationCoefficientDHW_BT2, outputVectorOptimization_chargingPowerEV_BT3, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT4,  outputVectorOptimization_chargingPowerBAT_BT5, outputVectorOptimization_disChargingPowerBAT_BT5 =   Building_Combined.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4,indexOfBuildingsOverall_BT5, currentDay)

            #Call the internal controller with the schedules
            overruleActions = True
            simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, negativeScore_total_overall = ICSimulation.simulateDays_WithAddtionalController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, overruleActions, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT1, outputVectorOptimization_heatGenerationCoefficientDHW_BT1, outputVectorOptimization_chargingPowerEV_BT1, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT2, outputVectorOptimization_heatGenerationCoefficientDHW_BT2, outputVectorOptimization_chargingPowerEV_BT3, outputVectorOptimization_heatGenerationCoefficientSpaceHeating_BT4, outputVectorOptimization_chargingPowerBAT_BT5, outputVectorOptimization_disChargingPowerBAT_BT5, pathForCreatingTheResultData_Centralized)
            if negativeScore_total_overall > 0.1:
                numberOfDaysWithNegativeScore = numberOfDaysWithNegativeScore + 1
            #Save results of the runs in a csv file 
            df_results_overall.loc[len(df_results_overall)] = [buildingIndex, currentDay, simulationObjective_surplusEnergy_kWh_combined[0], simulationObjective_maximumLoad_kW_combined[0], simulationObjective_costs_Euro_combined[0], simulationObjective_combinedScore_combined[0], negativeScore_total_overall[0]]
            df_results_overall.to_csv( pathForCreatingTheResultData_Centralized  + "/results_overall.csv", sep=";") 
                      
        print()
        print()
        print("Finished generateTrainingDataANN")
        print("numberOfDaysWithNegativeScore: ", numberOfDaysWithNegativeScore)

    


####################################################################################################################################################################################



#Method for randomly assigning days to the training and test data
def chooseTrainingAndTestDays_Random (numberOfTrainingDays_Overall, numberOfBuildingsForTrainingData_Overall, numberOfTestDays_Oveall, numberOfBuildingsForTestData_Overall, numberOfBuildingDataOverall):
    
    from random import randrange
    import numpy as np
    
    trainingDays_Overall = np.zeros((numberOfBuildingsForTrainingData_Overall, numberOfTrainingDays_Overall))
    testDays_Overall = np.zeros((numberOfBuildingsForTestData_Overall, numberOfTestDays_Oveall))

    usedTestData = np.zeros((numberOfBuildingDataOverall, 365))
    indexTestDay = 0
    indexBuilding = 0
    while indexBuilding < numberOfBuildingsForTestData_Overall:
        while indexTestDay < numberOfTestDays_Oveall:
            # Choose the test data
            randomNumber_DayOfTheYear = randrange(365)

            while randomNumber_DayOfTheYear > 89 and randomNumber_DayOfTheYear < 273:
                randomNumber_DayOfTheYear = randrange(365)
            if usedTestData[indexBuilding][randomNumber_DayOfTheYear] == 0:
                usedTestData[indexBuilding][randomNumber_DayOfTheYear] = 1
                testDays_Overall [indexBuilding][indexTestDay] = randomNumber_DayOfTheYear
            elif usedTestData[indexBuilding][randomNumber_DayOfTheYear] == 1:
                continue
            indexTestDay = indexTestDay + 1

        indexBuilding = indexBuilding + 1
        indexTestDay = 0

    #testDays_Overall= np.sort(testDays_Overall, axis=1).flatten()


    usedTrainingData = np.zeros((numberOfBuildingDataOverall, 365))
    indexTrainingDay = 0
    indexBuilding = 0
    while indexBuilding < numberOfBuildingsForTrainingData_Overall:
        while indexTrainingDay < numberOfTrainingDays_Overall:
            # Choose the Training data
            randomNumber_DayOfTheYear = randrange(365)

            while randomNumber_DayOfTheYear > 89 and randomNumber_DayOfTheYear < 275:
                randomNumber_DayOfTheYear = randrange(365)
            if usedTrainingData[indexBuilding][randomNumber_DayOfTheYear] == 0 and usedTestData [indexBuilding][randomNumber_DayOfTheYear] == 0:
                usedTrainingData[indexBuilding][randomNumber_DayOfTheYear] = 1
                trainingDays_Overall [indexBuilding][indexTrainingDay] = randomNumber_DayOfTheYear
            elif usedTrainingData[indexBuilding][randomNumber_DayOfTheYear] == 1 or usedTestData [indexBuilding][randomNumber_DayOfTheYear] == 1:
                continue
            indexTrainingDay = indexTrainingDay + 1

        indexBuilding = indexBuilding + 1
        indexTrainingDay = 0


    trainingDays_Overall = trainingDays_Overall.astype(int)
    testDays_Overall= testDays_Overall.astype(int)

    
    return trainingDays_Overall, testDays_Overall



#################################################################################################################################################################################################################

#Run simulations


if __name__ == "__main__":
    import Building_Combined
    import ANN
    import ICSimulation

    
    # define the directory to be created for the result files
    currentDatetimeString = datetime.today().strftime('%d_%m_%Y_Time_%H_%M_%S')
    simulationName = "Test"
    folderName_WholeSimulation = currentDatetimeString + "_" + simulationName + "_BTCombined_" + str(SetUpScenarios.numberOfBuildings_Total) 
    folderPath_WholeSimulation = "C:/Users/wi9632/Desktop/Ergebnisse/DSM/Instance_Base/" + folderName_WholeSimulation
    pathForCreatingTheResultData_Centralized = folderPath_WholeSimulation + "/Centralized"
    pathForCreatingTheResultData_Decentralized = folderPath_WholeSimulation + "/Decentralized"
    pathForCreatingTheResultData_SupervisedML = folderPath_WholeSimulation + "/ML"
    pathForCreatingTheResultData_RL = folderPath_WholeSimulation + "/RL"
    pathForCreatingTheResultData_Conventional = folderPath_WholeSimulation + "/Conventional"
    
   

    try:
        os.makedirs(folderPath_WholeSimulation)
        os.makedirs(pathForCreatingTheResultData_Centralized)    
        os.makedirs(pathForCreatingTheResultData_Decentralized)
        os.makedirs(pathForCreatingTheResultData_SupervisedML)
        os.makedirs(pathForCreatingTheResultData_RL)
        os.makedirs(pathForCreatingTheResultData_Conventional)
    except OSError:
        print ("Creation of the directory %s failed" % folderPath_WholeSimulation)
    else:
        print ("Successfully created the directory %s" % folderPath_WholeSimulation)


    if generateTrainingData ==True:
        generateTrainingDataForML()
    
    #Exact methods decentralized (testing)

    #Exact methods centralized (testing)
    if useCentralizedOptimization == True:
        print("\n-----------Centralized Optimization---------\n")
        
        currentDay = 1
        indexOfBuildingsOverall_BT1 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT1 + 1)]
        indexOfBuildingsOverall_BT2 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT2 + 1)]
        indexOfBuildingsOverall_BT3 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT3 + 1)]
        indexOfBuildingsOverall_BT4 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT4 + 1 )]
        indexOfBuildingsOverall_BT5 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT5 + 1 )]
        outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5 =   Building_Combined.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay)

        #Call the internal controller with the schedules
        overruleActions = False
        simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, negativeScore_total_overall = ICSimulation.simulateDays_WithAddtionalController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, overruleActions, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, pathForCreatingTheResultData_Centralized)


    #ANN methods (testing)
    if useSupervisedLearning == True:
        print("\n--------------Supervised Control------------\n")
        pathForTheTrainedModels = pathForCreatingTheResultData_SupervisedML + "/ML Training Configurations/"
        os.makedirs(pathForTheTrainedModels)
        currentDay = 1
        indexOfBuildingsOverall_BT1 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT1 + 1)]
        indexOfBuildingsOverall_BT2 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT2 + 1)]
        indexOfBuildingsOverall_BT3 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT3 + 1)]
        indexOfBuildingsOverall_BT4 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT4 + 1 )]
        indexOfBuildingsOverall_BT5 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT5 + 1 )]


        if SetUpScenarios.numberOfBuildings_BT1 >= 1:

            #Choose training and test days
            daySelectionMethod = 'Random'    # Options: ['Random'] ['Clustering_kMeans'] ['Clustering_Wards']
            numberOfTrainingDays = 80
            numberOfBuildingsForTrainingData_Overall = 10
            numberOfTestDays_Oveall = 20
            numberOfBuildingsForTestData_Overall = 5
            trainingDaysForSupervisedLearning, testDaysForSupvervisedControl = chooseTrainingAndTestDays_Random(numberOfTrainingDays, numberOfBuildingsForTrainingData_Overall , numberOfTestDays_Oveall ,numberOfBuildingsForTestData_Overall, numberOfBuildingDataOverall)

            #TEST Clustering
            ANN.clusterTrainingData(trainingDaysForSupervisedLearning, 0, 0)

            #Train the supvervised learning model
            usedMLMethod = ML_METHOD_RANDOM_FOREST  # Options: [ML_METHOD_MULTI_LAYER_PERCEPTRON], [ML_METHOD_RANDOM_FOREST], [ML_METHOD_GRADIENT_BOOSTING], [ML_METHOD_RNN], [ML_METHOD_RNN]
            objective = OPT_OBJECTIVE_MIN_SURPLUS   # Options: [OPT_OBJECTIVE_MIN_SURPLUS], [OPT_OBJECTIVE_MIN_PEAK], [OPT_OBJECTIVE_MIN_COSTS]
            useNormalizedData = False
            useStandardizedData = True
            dayClusterName = 'ClusterAllDays'
            practiseModeWithTestPredictions = True
            perfectForecastForSequencePredictions = False



            #Call method for training the supervised ML
            if usedMLMethod == ML_METHOD_MULTI_LAYER_PERCEPTRON or usedMLMethod == ML_METHOD_RANDOM_FOREST or usedMLMethod == ML_METHOD_GRADIENT_BOOSTING:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainSupervisedML_SingleTimeslot_SingleBuildingOptScenario (trainingDaysForSupervisedLearning, objective ,useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName,pathForTheTrainedModels, practiseModeWithTestPredictions)

                #Call method for the simulation of one day by generating and taking actions for single time slots
                outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1, outputVectorANN_heatGenerationCoefficientDHW_BT1, outputVectorANN_chargingPowerEV_BT1, outputVectorANN_temperatureBufferStorage_BT1, outputVectorANN_volumeDHWTank_BT1, outputVectorANN_SOC_BT1 = ANN.generateActionsForSingleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            if usedMLMethod == ML_METHOD_LSTM or usedMLMethod == ML_METHOD_RNN:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainRNN_MultipleTimeslot_SingleBuildingOptScenario(trainingDaysForSupervisedLearning, objective, useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName, practiseModeWithTestPredictions, perfectForecastForSequencePredictions)

                # Call method for the simulation of one day by generating and taking actions for multiple time slots
                outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1, outputVectorANN_heatGenerationCoefficientDHW_BT1, outputVectorANN_chargingPowerEV_BT1 = ANN.generateActionsForMutipleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            #Set the irrelevant output vectors for this building to 0
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientDHW_BT2 =np.zeros(0)
            outputVectorANN_chargingPowerEV_BT3 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4 = np.zeros(0)
            outputVectorANN_chargingPowerBAT_BT5 = np.zeros(0)
            outputVectorANN_disChargingPowerEV_BT5 = np.zeros(0)

            #Reshape outputdata of the ANN (:=input data for the internal controller)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1 = outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1.reshape((1, SetUpScenarios.numberOfTimeSlotsPerDay), order='F')
            outputVectorANN_heatGenerationCoefficientDHW_BT1 = outputVectorANN_heatGenerationCoefficientDHW_BT1.reshape((1, SetUpScenarios.numberOfTimeSlotsPerDay), order='F')
            outputVectorANN_chargingPowerEV_BT1 = outputVectorANN_chargingPowerEV_BT1.reshape((1, SetUpScenarios.numberOfTimeSlotsPerDay), order='F')

            #Call the internal controller with the schedules
            overruleActions = False
            simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, negativeScore_total_overall = ICSimulation.simulateDays_WithAddtionalController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, overruleActions, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1, outputVectorANN_heatGenerationCoefficientDHW_BT1, outputVectorANN_chargingPowerEV_BT1, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2, outputVectorANN_heatGenerationCoefficientDHW_BT2, outputVectorANN_chargingPowerEV_BT3, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4, outputVectorANN_chargingPowerBAT_BT5, outputVectorANN_disChargingPowerEV_BT5, pathForCreatingTheResultData_SupervisedML)


        if SetUpScenarios.numberOfBuildings_BT2 == 1:
            #Choose training and test days
            daySelectionMethod = 'Random'    # Options: ['Random'] ['Clustering_kMeans'] ['Clustering_Wards']
            numberOfTrainingDays = 80
            numberOfBuildingsForTrainingData_Overall = 3
            numberOfTestDays_Oveall = 20
            numberOfBuildingsForTestData_Overall = 1
            trainingDaysForSupervisedLearning, testDaysForSupvervisedControl = chooseTrainingAndTestDays_Random(numberOfTrainingDays, numberOfBuildingsForTrainingData_Overall , numberOfTestDays_Oveall ,numberOfBuildingsForTestData_Overall, numberOfBuildingDataOverall)

            #Train the supvervised learning model
            usedMLMethod = ML_METHOD_GRADIENT_BOOSTING  # Options: [ML_METHOD_MULTI_LAYER_PERCEPTRON], [ML_METHOD_RANDOM_FOREST], [ML_METHOD_GRADIENT_BOOSTING], [ML_METHOD_RNN], [ML_METHOD_RNN]
            objective = OPT_OBJECTIVE_MIN_PEAK   # Options: [OPT_OBJECTIVE_MIN_SURPLUS], [OPT_OBJECTIVE_MIN_PEAK], [OPT_OBJECTIVE_MIN_COSTS]
            useNormalizedData = False
            useStandardizedData = True
            dayClusterName = 'ClusterAllDays'
            practiseModeWithTestPredictions = True
            perfectForecastForSequencePredictions = False


            #Call method for training the supervised ML
            if usedMLMethod == ML_METHOD_MULTI_LAYER_PERCEPTRON or usedMLMethod == ML_METHOD_RANDOM_FOREST or usedMLMethod == ML_METHOD_GRADIENT_BOOSTING:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainSupervisedML_SingleTimeslot_SingleBuildingOptScenario (trainingDaysForSupervisedLearning, objective ,useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName,pathForTheTrainedModels, practiseModeWithTestPredictions)

                #Call method for the simulation of one day by generating and taking actions for single time slots
                outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2, outputVectorANN_heatGenerationCoefficientDHW_BT2, outputVectorANN_temperatureBufferStorage_BT2, outputVectorANN_volumeDHWTank_BT2 = ANN.generateActionsForSingleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            if usedMLMethod == ML_METHOD_LSTM or usedMLMethod == ML_METHOD_RNN:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainRNN_MultipleTimeslot_SingleBuildingOptScenario(trainingDaysForSupervisedLearning, objective, useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName, practiseModeWithTestPredictions, perfectForecastForSequencePredictions)

                # Call method for the simulation of one day by generating and taking actions for multiple time slots
                outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2, outputVectorANN_heatGenerationCoefficientDHW_BT2 = ANN.generateActionsForMutipleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            #Call method for the simulation of one day by generating and taking actions for single time slots

            #Set the irrelevant output vectors for this building to 0
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientDHW_BT1 = np.zeros(0)
            outputVectorANN_chargingPowerEV_BT1 = np.zeros(0)
            outputVectorANN_chargingPowerEV_BT3 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4 = np.zeros(0)
            outputVectorANN_chargingPowerBAT_BT5 = np.zeros(0)
            outputVectorANN_disChargingPowerEV_BT5 = np.zeros(0)

            #Reshape outputdata of the ANN (:=input data for the internal controller)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2 = outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2.reshape((1, SetUpScenarios.numberOfTimeSlotsPerDay), order='F')
            outputVectorANN_heatGenerationCoefficientDHW_BT2 = outputVectorANN_heatGenerationCoefficientDHW_BT2.reshape((1, SetUpScenarios.numberOfTimeSlotsPerDay), order='F')


            #Call the internal controller with the schedules
            overruleActions = False
            simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, negativeScore_total_overall = ICSimulation.simulateDays_WithAddtionalController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, overruleActions, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1, outputVectorANN_heatGenerationCoefficientDHW_BT1, outputVectorANN_chargingPowerEV_BT1, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2, outputVectorANN_heatGenerationCoefficientDHW_BT2, outputVectorANN_chargingPowerEV_BT3, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4, outputVectorANN_chargingPowerBAT_BT5, outputVectorANN_disChargingPowerEV_BT5, pathForCreatingTheResultData_SupervisedML)

        if SetUpScenarios.numberOfBuildings_BT3 == 1:
            #Choose training and test days
            daySelectionMethod = 'Random'    # Options: ['Random'] ['Clustering_kMeans'] ['Clustering_Wards']
            numberOfTrainingDays = 80
            numberOfBuildingsForTrainingData_Overall = 2
            numberOfTestDays_Oveall = 20
            numberOfBuildingsForTestData_Overall = 1
            trainingDaysForSupervisedLearning, testDaysForSupvervisedControl = chooseTrainingAndTestDays_Random(numberOfTrainingDays, numberOfBuildingsForTrainingData_Overall , numberOfTestDays_Oveall ,numberOfBuildingsForTestData_Overall, numberOfBuildingDataOverall)

            #Train the supvervised learning model
            usedMLMethod = ML_METHOD_RANDOM_FOREST  # Options: [ML_METHOD_MULTI_LAYER_PERCEPTRON], [ML_METHOD_RANDOM_FOREST], [ML_METHOD_GRADIENT_BOOSTING], [ML_METHOD_RNN], [ML_METHOD_RNN]
            objective = OPT_OBJECTIVE_MIN_COSTS   # Options: [OPT_OBJECTIVE_MIN_SURPLUS], [OPT_OBJECTIVE_MIN_PEAK], [OPT_OBJECTIVE_MIN_COSTS]
            useNormalizedData = False
            useStandardizedData = True
            dayClusterName = 'ClusterAllDays'
            practiseModeWithTestPredictions = True
            perfectForecastForSequencePredictions = False


            #Call method for training the supervised ML
            if usedMLMethod == ML_METHOD_MULTI_LAYER_PERCEPTRON or usedMLMethod == ML_METHOD_RANDOM_FOREST or usedMLMethod == ML_METHOD_GRADIENT_BOOSTING:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainSupervisedML_SingleTimeslot_SingleBuildingOptScenario (trainingDaysForSupervisedLearning, objective ,useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName,pathForTheTrainedModels, practiseModeWithTestPredictions)

                #Call method for the simulation of one day by generating and taking actions for single time slots
                outputVectorANN_chargingPowerEV_BT3, outputVectorANN_SOC_BT3 = ANN.generateActionsForSingleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            if usedMLMethod == ML_METHOD_LSTM or usedMLMethod == ML_METHOD_RNN:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainRNN_MultipleTimeslot_SingleBuildingOptScenario(trainingDaysForSupervisedLearning, objective, useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName, practiseModeWithTestPredictions, perfectForecastForSequencePredictions)

                # Call method for the simulation of one day by generating and taking actions for multiple time slots
                outputVectorANN_chargingPowerEV_BT3 = ANN.generateActionsForMutipleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientDHW_BT1 = np.zeros(0)
            outputVectorANN_chargingPowerEV_BT1 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientDHW_BT2 =np.zeros(0)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4 = np.zeros(0)
            outputVectorANN_chargingPowerBAT_BT5 = np.zeros(0)
            outputVectorANN_disChargingPowerEV_BT5 = np.zeros(0)

            #Reshape outputdata of the ANN (:=input data for the internal controller)
            outputVectorANN_chargingPowerEV_BT3 = outputVectorANN_chargingPowerEV_BT3.reshape((1, SetUpScenarios.numberOfTimeSlotsPerDay), order='F')

            #Call the internal controller with the schedules
            overruleActions = False
            simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, negativeScore_total_overall = ICSimulation.simulateDays_WithAddtionalController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, overruleActions, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1, outputVectorANN_heatGenerationCoefficientDHW_BT1, outputVectorANN_chargingPowerEV_BT1, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2, outputVectorANN_heatGenerationCoefficientDHW_BT2, outputVectorANN_chargingPowerEV_BT3, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4, outputVectorANN_chargingPowerBAT_BT5, outputVectorANN_disChargingPowerEV_BT5, pathForCreatingTheResultData_SupervisedML)

        if SetUpScenarios.numberOfBuildings_BT4 == 1:
            #Choose training and test days
            daySelectionMethod = 'Random'    # Options: ['Random'] ['Clustering_kMeans'] ['Clustering_Wards']
            numberOfTrainingDays = 80
            numberOfBuildingsForTrainingData_Overall = 2
            numberOfTestDays_Oveall = 20
            numberOfBuildingsForTestData_Overall = 1
            trainingDaysForSupervisedLearning, testDaysForSupvervisedControl = chooseTrainingAndTestDays_Random(numberOfTrainingDays, numberOfBuildingsForTrainingData_Overall , numberOfTestDays_Oveall ,numberOfBuildingsForTestData_Overall, numberOfBuildingDataOverall)

            #Train the supvervised learning model
            usedMLMethod = ML_METHOD_RANDOM_FOREST  # Options: [ML_METHOD_MULTI_LAYER_PERCEPTRON], [ML_METHOD_RANDOM_FOREST], [ML_METHOD_GRADIENT_BOOSTING], [ML_METHOD_RNN], [ML_METHOD_RNN]
            objective = OPT_OBJECTIVE_MIN_SURPLUS   # Options: [OPT_OBJECTIVE_MIN_SURPLUS], [OPT_OBJECTIVE_MIN_PEAK], [OPT_OBJECTIVE_MIN_COSTS]
            useNormalizedData = False
            useStandardizedData = True
            dayClusterName = 'ClusterAllDays'
            practiseModeWithTestPredictions = True
            perfectForecastForSequencePredictions = False


            #Call method for training the supervised ML
            if usedMLMethod == ML_METHOD_MULTI_LAYER_PERCEPTRON or usedMLMethod == ML_METHOD_RANDOM_FOREST or usedMLMethod == ML_METHOD_GRADIENT_BOOSTING:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainSupervisedML_SingleTimeslot_SingleBuildingOptScenario (trainingDaysForSupervisedLearning, objective ,useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName,pathForTheTrainedModels, practiseModeWithTestPredictions)

                #Call method for the simulation of one day by generating and taking actions for single time slots
                outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4, outputVectorANN_temperatureBufferStorage_BT4 = ANN.generateActionsForSingleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            if usedMLMethod == ML_METHOD_LSTM or usedMLMethod == ML_METHOD_RNN:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainRNN_MultipleTimeslot_SingleBuildingOptScenario(trainingDaysForSupervisedLearning, objective, useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName, practiseModeWithTestPredictions, perfectForecastForSequencePredictions)

                # Call method for the simulation of one day by generating and taking actions for multiple time slots
                outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4 = ANN.generateActionsForMutipleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientDHW_BT1 = np.zeros(0)
            outputVectorANN_chargingPowerEV_BT1 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientDHW_BT2 =np.zeros(0)
            outputVectorANN_chargingPowerEV_BT3 = np.zeros(0)
            outputVectorANN_chargingPowerBAT_BT5 = np.zeros(0)
            outputVectorANN_disChargingPowerEV_BT5 = np.zeros(0)


            #Reshape outputdata of the ANN (:=input data for the internal controller)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4 = outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4.reshape((1, SetUpScenarios.numberOfTimeSlotsPerDay), order='F')

            #Call the internal controller with the schedules
            overruleActions = False
            simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, negativeScore_total_overall = ICSimulation.simulateDays_WithAddtionalController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, overruleActions, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1, outputVectorANN_heatGenerationCoefficientDHW_BT1, outputVectorANN_chargingPowerEV_BT1, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2, outputVectorANN_heatGenerationCoefficientDHW_BT2, outputVectorANN_chargingPowerEV_BT3, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4, outputVectorANN_chargingPowerBAT_BT5, outputVectorANN_disChargingPowerEV_BT5, pathForCreatingTheResultData_SupervisedML)

        if SetUpScenarios.numberOfBuildings_BT5 == 1:
            #Choose training and test days
            daySelectionMethod = 'Random'    # Options: ['Random'] ['Clustering_kMeans'] ['Clustering_Wards']
            numberOfTrainingDays = 80
            numberOfBuildingsForTrainingData_Overall = 2
            numberOfTestDays_Oveall = 20
            numberOfBuildingsForTestData_Overall = 1
            trainingDaysForSupervisedLearning, testDaysForSupvervisedControl = chooseTrainingAndTestDays_Random(numberOfTrainingDays, numberOfBuildingsForTrainingData_Overall , numberOfTestDays_Oveall ,numberOfBuildingsForTestData_Overall, numberOfBuildingDataOverall)

            #Train the supvervised learning model
            usedMLMethod = ML_METHOD_GRADIENT_BOOSTING  # Options: [ML_METHOD_MULTI_LAYER_PERCEPTRON], [ML_METHOD_RANDOM_FOREST], [ML_METHOD_GRADIENT_BOOSTING], [ML_METHOD_RNN], [ML_METHOD_RNN]
            objective = OPT_OBJECTIVE_MIN_COSTS   # Options: [OPT_OBJECTIVE_MIN_SURPLUS], [OPT_OBJECTIVE_MIN_PEAK], [OPT_OBJECTIVE_MIN_COSTS]
            useNormalizedData = False
            useStandardizedData = True
            dayClusterName = 'ClusterAllDays'
            practiseModeWithTestPredictions = True
            perfectForecastForSequencePredictions = False

            #Call method for training the supervised ML
            if usedMLMethod == ML_METHOD_MULTI_LAYER_PERCEPTRON or usedMLMethod == ML_METHOD_RANDOM_FOREST or usedMLMethod == ML_METHOD_GRADIENT_BOOSTING:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainSupervisedML_SingleTimeslot_SingleBuildingOptScenario (trainingDaysForSupervisedLearning, objective ,useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName,pathForTheTrainedModels, practiseModeWithTestPredictions)

                #Call method for the simulation of one day by generating and taking actions for single time slots
                outputVectorANN_chargingPowerBAT_BT5, outputVectorANN_disChargingPowerBAT_BT5, outputVectorANN_SOC_BAT_BT5 = ANN.generateActionsForSingleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            if usedMLMethod == ML_METHOD_LSTM or usedMLMethod == ML_METHOD_RNN:
                dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel = ANN.trainRNN_MultipleTimeslot_SingleBuildingOptScenario(trainingDaysForSupervisedLearning, objective, useNormalizedData, useStandardizedData, usedMLMethod, dayClusterName, practiseModeWithTestPredictions, perfectForecastForSequencePredictions)

                # Call method for the simulation of one day by generating and taking actions for multiple time slots
                outputVectorANN_chargingPowerBAT_BT5, outputVectorANN_disChargingPowerBAT_BT5 = ANN.generateActionsForMutipleTimeslotWithANN_SingleBuildingOptScenario(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_SupervisedML,objective, daySelectionMethod, dataScaler_InputFeatures, dataScaler_OutputLabels, trainedModel)


            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientDHW_BT1 = np.zeros(0)
            outputVectorANN_chargingPowerEV_BT1 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientDHW_BT2 =np.zeros(0)
            outputVectorANN_chargingPowerEV_BT3 = np.zeros(0)
            outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4 = np.zeros(0)

            #Reshape outputdata of the ANN (:=input data for the internal controller)
            outputVectorANN_chargingPowerBAT_BT5 = outputVectorANN_chargingPowerBAT_BT5.reshape((1, SetUpScenarios.numberOfTimeSlotsPerDay), order='F')
            outputVectorANN_disChargingPowerBAT_BT5 = outputVectorANN_disChargingPowerBAT_BT5.reshape((1, SetUpScenarios.numberOfTimeSlotsPerDay), order='F')

            #Call the internal controller with the schedules
            overruleActions = False
            simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, negativeScore_total_overall = ICSimulation.simulateDays_WithAddtionalController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, currentDay, overruleActions, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT1, outputVectorANN_heatGenerationCoefficientDHW_BT1, outputVectorANN_chargingPowerEV_BT1, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT2, outputVectorANN_heatGenerationCoefficientDHW_BT2, outputVectorANN_chargingPowerEV_BT3, outputVectorANN_heatGenerationCoefficientSpaceHeating_BT4, outputVectorANN_chargingPowerBAT_BT5, outputVectorANN_disChargingPowerBAT_BT5, pathForCreatingTheResultData_SupervisedML)



    
    # RL methods
    if useReinforcementLearning == True:
        print("RL Control")     
      
        
    # Conventional Control
    if useConventionalControl == True:
        print("\n--------------Conventional Control------------\n")
        currentDay = 1
        indexOfBuildingsOverall_BT1 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT1)]
        indexOfBuildingsOverall_BT2 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT2)]
        indexOfBuildingsOverall_BT3 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT3)]
        indexOfBuildingsOverall_BT4 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT4)]
        indexOfBuildingsOverall_BT5 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT5)]

        simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, negativeScore_total_overall = ICSimulation.simulateDays_ConventionalControl(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4,indexOfBuildingsOverall_BT5, currentDay, pathForCreatingTheResultData_Conventional)










    

