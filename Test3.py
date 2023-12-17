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
    3416.91, 1383.88, 1551.66, 4324.61, 969.77, 431.08, 703.39, 1201.25, 898.28, 4969.99,
    7839.21, 259.08, 410.82, 617.37, 661.68, 552.18, 319.37, 702.73
])

# Convert electricity load values to kW
electricity_load_kW = electricity_load / 1000

# Price values in Cent/kWh
price_values = [
    27.1,  26.5,  26.0, 25.1, 26.6, 27.5,  34.4, 51.3, 45.3,  44.3, 44.3,  41.3,  38.1,  35.5,
    33.9,  37, 41.4,  48.6,  53.4,  48.6,  43.4, 38.7, 37.8,  27.4,
]

# Duplicate the last entry for a smooth horizontal line at the end
time_of_day.append("23:00")
price_values.append(price_values[-1])

# Create subplots within the same figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plotting the bar diagram
ax1.bar(time_of_day, electricity_load_kW[:len(time_of_day)], color='#FFD700', edgecolor='black')  # Dark yellow color

# Labeling the first plot
ax1.set_ylabel('Electricity Load (kW)', fontsize=18)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plotting the step line plot
ax2.step(time_of_day, price_values, where='post', color='limegreen', linewidth=6)  # Adjusted where parameter

# Labeling the second plot
ax2.set_xlabel('Time of Day', fontsize=18)
ax2.set_ylabel('Price (Cent/kWh)', fontsize=18)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust x-axis limits to eliminate empty space
ax1.set_xlim(time_of_day[0], time_of_day[-1])

# Increase thickness and rotation of x-tick labels
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', which='both', labelsize=14, width=2, rotation=45, ha='right')

# Automatically adjust subplot parameters for better spacing
plt.tight_layout()

# Save the figure
plt.savefig('C:/Users/wi9632/Desktop/temp_combined.png', dpi=100)

# Show the figure
plt.show()
