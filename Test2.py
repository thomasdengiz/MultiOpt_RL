import os
import pandas as pd
import matplotlib.pyplot as plt

# Define folder paths
base_path = "C:\\Users\\wi9632\\Desktop\\Results_End_MultiOpt"
output_folder = "C:\\Users\\wi9632\\Desktop\\Pictures"

# Define method names and colors
methods = {
    "Base_Min10_BTCombined_30_Run2": ("PALSS", "tab:orange"),
    "NSGAII_Min10_BTCombined_30_Run2": ("NSGA-II", "tab:purple"),
    "RL_Min10_BTCombined_30_Run2": ("RELAPALSS", "tab:cyan"),
    "SPEAII_Min10_BTCombined_30_Run2": ("SPEA-II", "gold"),
    "ConventionalControl_BTCombined_30": ("Conventional Control", "red"),
    "DichotomousMethod_BTCombined_30": ("Dichotomous Method", "lawngreen")
}


# Function to plot data for each Day subfolder
def plot_day(day_folder, method_data):
    plt.figure(figsize=(10, 8))  # Increase figure size
    for method, data in method_data.items():
        if data is not None:
            plt.scatter(data["Costs"], data["Peak Load"], label=methods[method][0], color=methods[method][1], alpha=0.7,
                        s=180, edgecolors='black', linewidths=0.5)  # Add black edge around each point
    plt.xlabel("Costs [€]", fontsize=22)  # Increase size of x-label
    plt.ylabel("Peak Load [kW]", fontsize=22)  # Increase size of y-label

    # Separate "Day" and number with space in title
    day_number = "Unknown"
    if "Day" in day_folder:
        day_number = day_folder.split("Day")[1]

    plt.title(f"30 buildings - Solutions for Day {day_number}", fontsize=24)
    plt.legend(loc="upper left", fontsize=15)
    plt.grid(True, alpha=0.5)
    plt.xticks(fontsize=20)  # Increase size of x-label ticks
    plt.yticks(fontsize=20)  # Increase size of y-label ticks
    plt.savefig(os.path.join(output_folder, f"{day_folder}.png"), dpi=200)
    plt.close()


# Function to clean values with brackets
def clean_value(value):
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        return float(value[1:-1])  # Remove brackets and convert to float
    return value


# Traverse through day folders
for day_folder in os.listdir(
        os.path.join(base_path, next(iter(methods.keys())))):  # Pick any method to get the day folders
    method_data = {}
    for method, (name, _) in methods.items():
        method_folder = os.path.join(base_path, method, day_folder)
        if os.path.isdir(method_folder):
            csv_files = [file for file in os.listdir(method_folder) if
                         file.startswith("ParetoFront") and file.endswith(".csv")]
            if csv_files:
                method_data[method] = pd.concat(
                    [pd.read_csv(os.path.join(method_folder, file), delimiter=";").applymap(clean_value) for file in
                     csv_files],
                    ignore_index=True)
            else:
                method_data[method] = None
        else:
            method_data[method] = None

    # Plot data for the Day subfolder
    plot_day(day_folder, method_data)

