import matplotlib.pyplot as plt
import numpy as np

# Time of day values
time_of_day = [
    "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00",
    "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
]

# Electricity load values in W
electricity_load = np.array([
    5460.99, 177.71, 163.63, 745.47, 770.04, 1049.45, 1090.48, 868.8, 277.68, 3117.22,
    3416.91, 1383.88, 1551.66, 7324.61, 969.77, 431.08, 703.39, 1201.25, 898.28, 4969.99,
    5839.21, 259.08, 410.82, 617.37, 661.68, 552.18, 319.37, 702.73
])

# Convert electricity load values to kW
electricity_load_kW = electricity_load / 1000

# Increase the figure size for better visibility
plt.figure(figsize=(12, 6))

# Plotting the bar diagram
plt.bar(time_of_day, electricity_load_kW[:len(time_of_day)], color='#FFD700', edgecolor='black')  # Dark yellow color

# Adjusting x-axis ticks
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=16)

# Labeling the plot
plt.xlabel('Time of Day', fontsize=18)
plt.ylabel('Electricity Load (kW)', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set x-axis limits to remove extra space on the left and right
plt.xlim(-0.5, len(time_of_day) - 0.5)

# Automatically adjust subplot parameters for better spacing
plt.tight_layout()

plt.savefig('C:/Users/wi9632/Desktop/temp.png', dpi=100)

# Show the plot
plt.show()



