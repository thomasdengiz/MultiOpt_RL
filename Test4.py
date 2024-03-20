import matplotlib.pyplot as plt
import numpy as np

# Data
algorithms = ['PALSS', 'RELAPALSS', 'NSGA-II', 'SPEA-II']
categories = ['10 buildings', '20 buildings', '30 buildings']
shift_GD = np.array([
    [9.1, 23.4, 37.3],
    [8.7, 23.1, 37.1],
    [16.2, 31.6, 46.9],
    [13.8, 26.4, 43.0]
])
shift_HV = np.array([
    [122.4, 214.9, 312.8],
    [125.1, 216.8, 313.5],
    [20.4, 43.6, 90.6],
    [39.3, 105.4, 141.4]
])

# Define colors
colors = ['tab:orange', 'tab:cyan', 'tab:purple', 'gold']

# Plotting Shift GD
plt.figure(figsize=(12, 6))
bar_width = 0.2
index = np.arange(len(categories)) * 1.2  # Adjusted index spacing

for i, algo in enumerate(algorithms):
    plt.bar(index + i * bar_width, shift_GD[i], bar_width, label=algo, color=colors[i])


plt.ylabel('GD Value', fontsize=15)
plt.xticks(index + bar_width * 1.5, categories, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(algorithms), fontsize=17)

# Adding values on top of bars
for i, idx in enumerate(index):
    for j, algo in enumerate(algorithms):
        plt.text(idx + j * bar_width, shift_GD[j][i] + 0.5, str(shift_GD[j][i]), ha='center', fontsize=13)

plt.tight_layout()
plt.savefig('C:/Users/wi9632/Desktop/GD_combined.png', dpi=200)
plt.show()

# Plotting Shift HV
plt.figure(figsize=(12, 6))

for i, algo in enumerate(algorithms):
    plt.bar(index + i * bar_width, shift_HV[i], bar_width, label=algo, color=colors[i])


plt.ylabel('HV Value', fontsize=15)
plt.xticks(index + bar_width * 1.5, categories, fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, 340)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(algorithms), fontsize=17)

# Adding values on top of bars
for i, idx in enumerate(index):
    for j, algo in enumerate(algorithms):
        plt.text(idx + j * bar_width, shift_HV[j][i] + 5, str(shift_HV[j][i]), ha='center', fontsize=13)

plt.tight_layout()
plt.savefig('C:/Users/wi9632/Desktop/HV_combined.png', dpi=200)
plt.show()


