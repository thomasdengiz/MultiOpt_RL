import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define folder path
pictures_folder = "C:\\Users\\wi9632\\Desktop\\Pictures"

# Define method names and colors
methods = {
    "Base_Min10_BTCombined_30_Run2": ("PALSS", "tab:orange"),
    "NSGAII_Min10_BTCombined_30_Run2": ("NSGA-II", "tab:purple"),
    "RL_Min10_BTCombined_30_Run2": ("RELAPALSS", "tab:cyan"),
    "SPEAII_Min10_BTCombined_30_Run2": ("SPEA-II", "gold"),
    "ConventionalControl_BTCombined_30": ("Conventional Control", "red"),
    "DichotomousMethod_BTCombined_30": ("Dichotomous Method", "lawngreen")
}

# Create a figure to hold the plots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust spacing between subplots

# Iterate over the PNG files and plot them
for idx, file_name in enumerate(["BT10_Day11.png", "BT10_Day45.png", "BT10_Day361.png",
                                 "BT20_Day15.png", "BT20_Day332.png", "BT20_Day346.png",
                                 "BT30_Day39.png", "BT30_Day80.png", "BT30_Day292.png"]):
    row = idx // 3
    col = idx % 3
    img = plt.imread(os.path.join(pictures_folder, file_name))
    axes[row, col].imshow(img)
    axes[row, col].axis('off')

# Add legend at the top of the combined plot
legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markersize=10, markerfacecolor=color)
                   for name, color in methods.values()]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=len(methods))

# Save the plot
plt.savefig(os.path.join(pictures_folder, "ParetoFronts_combined.png"), dpi=200, bbox_inches='tight')
plt.show()

