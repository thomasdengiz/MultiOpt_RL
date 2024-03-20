import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("C:/Users/wi9632/Desktop/Load Profiles One Day.csv", sep=";", encoding='ISO-8859-1')

# Convert each column to a NumPy array
timeslots = np.array(df['timeslot'])
time_of_day = np.array(df['time of day'])
day = np.array(df['Day'])
price = np.array(df['Price [Cent/kWh]'])
outside_temperature = np.array(df['Outside Temperature [C]'])
demand_space_heating = np.array(df['Demand Space Heating [W]'])
temperature_psc_ann = np.array(df['Temperature PSC-ANN [°C]'])
heating_activity_psc_ann = np.array(df['Heating Activity PSC-ANN'])
electrical_power_hp_psc_ann = np.array(df['Electrical power HP PSC-ANN [W]'])

temperature_psc = np.array(df['Temperature PSC [°C]'])
heating_activity_psc = np.array(df['Heating Activity PSC'])
electrical_power_hp_psc = np.array(df['Electrical power HP PSC [W]'])

temperature_cc = np.array(df['Temperature CC [°C]'])
heating_activity_cc = np.array(df['Heating Activity CC'])
electrical_power_hp_cc = np.array(df['Electrical power HP CC [W]'])

temperature_opt = np.array(df['Temperature Opt [°C]'])
heating_activity_opt = np.array(df['Heating Activity Opt'])
electrical_power_hp_opt = np.array(df['Electrical power HP Opt [W]'])

# Divide values by 1000 to get kW
electrical_power_hp_psc_ann_kW = electrical_power_hp_psc_ann / 1000
electrical_power_hp_psc_kW = electrical_power_hp_psc / 1000
electrical_power_hp_cc_kW = electrical_power_hp_cc / 1000
electrical_power_hp_opt_kW = electrical_power_hp_opt / 1000

# Create subplots within the same figure
fig, axs = plt.subplots(5, 1, figsize=(9, 11), sharex=True)

# Plotting the line charts
axs[0].plot(timeslots, temperature_psc_ann, label='PSC-ANN', color='blue')
axs[0].plot(timeslots, temperature_psc, label='PSC', color='green')
axs[0].plot(timeslots, temperature_cc, label='Conventional Control', color='orange')
axs[0].plot(timeslots, temperature_opt, label='Optimal Control', color='red')
axs[0].set_ylabel('Temperature (°C)', fontsize=11)
axs[0].set_ylim(19, 25)

# Plotting the area charts for Electrical Power (kW) without stacking
axs[1].plot(timeslots, electrical_power_hp_psc_ann_kW, label='PSC-ANN', color='blue', alpha=0.7)
axs[1].plot(timeslots, electrical_power_hp_psc_kW, label='PSC', color='green', alpha=0.7)
axs[1].plot(timeslots, electrical_power_hp_cc_kW, label='Conventional Control', color='orange', alpha=0.7)
axs[1].plot(timeslots, electrical_power_hp_opt_kW, label='Optimal Control', color='red', alpha=0.7)
axs[1].fill_between(timeslots, electrical_power_hp_psc_ann_kW, alpha=0.3, color='blue')
axs[1].fill_between(timeslots, electrical_power_hp_psc_kW, alpha=0.3, color='green')
axs[1].fill_between(timeslots, electrical_power_hp_cc_kW, alpha=0.3, color='orange')
axs[1].fill_between(timeslots, electrical_power_hp_opt_kW, alpha=0.3, color='red')
axs[1].set_ylabel('Electrical Power (kW)', fontsize=11)

# Plotting the line chart for Demand Space Heating (kW)
axs[2].plot(timeslots, demand_space_heating / 1000, color='grey')
axs[2].set_ylabel('Heat demand (kW)', fontsize=11)
axs[2].set_ylim(0, 40)

# Plotting the line chart for Outside Temperature (°C)
axs[3].plot(timeslots, outside_temperature, color='grey')
axs[3].set_ylabel('Outside temperature (°C)', fontsize=11)

# Plotting the line chart for Price (Cent/kWh)
axs[4].plot(timeslots, price, color='grey')
axs[4].set_xlabel('Weekday', fontsize=9)
axs[4].set_ylabel('Price (Cent/kWh)', fontsize=11)

# Add legend above the first plot
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4, fontsize=12)

# Set x-tick labels with rotation and horizontal alignment
weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_positions = [0, 48, 2*48, 3*48, 4*48, 5*48, 6*48]
plt.xticks(weekday_positions, weekday_labels, rotation=45, ha='right')

# Remove empty space on the x-axis
plt.xlim(timeslots.min() - 1, timeslots.max())

# Adjust the layout to make space for the legend
plt.subplots_adjust(top=0.9, bottom=0.1)

plt.savefig('C:/Users/wi9632/Desktop/Load_profiles_ANN_for_DSM_2.png', dpi=100)
# Show the plot
plt.show()

# Add a light grid
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.8)
