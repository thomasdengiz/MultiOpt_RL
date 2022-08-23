from joblib import dump, load
import numpy as np

dataScaler_InputFeatures = load("C:/Users/wi9632/Desktop/Ergebnisse/DSM/Instance_Base/21_07_2022_Time_10_08_13_Test_BTCombined_1/ML/ML Training Configurations/dataScalerInputeFeatures.bin")

vector_input_features =  np.array([1, 22, 150, 45, 0, 1000, 0, 432, 1, 13.5, 12, 2])
vector_input_features_reshaped = vector_input_features.reshape(1, -1)
vector_input_features_reshaped2 = vector_input_features.reshape(-1, 1)
vector_input_features_ravel = vector_input_features.ravel()
vector_input_features_flatten = vector_input_features.flatten()

vector_input_features_scaled = dataScaler_InputFeatures.transform (vector_input_features_reshaped)

print(f"vector_input_features_scaled: {vector_input_features_scaled}")
