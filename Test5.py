import pandas as pd
import matplotlib.pyplot as plt


file_path = r"C:\Users\wi9632\Desktop\Results_End_MultiOpt\Load_Profile_Figure\Combined_Table.csv"
df = pd.read_csv(file_path, sep=';', encoding='latin-1')



methods = {
    "Dichotomous Method [kW]": ("Dichotomous Method", "lawngreen"),
    "Conventional Control [kW]": ("Conventional Control", "red"),
    "NSGA-II [kW]": ("NSGA-II", "tab:purple"),
    "SPEA-II [kW]": ("SPEA-II", "gold"),
    "PALSS [kW]": ("PALSS", "tab:orange"),
    "RELAPALLS [kW]": ("RELAPALLS", "tab:cyan")
}


fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

# Plotting the first subplot with 6 lines and adding legend
for method, (name, color) in methods.items():
    axes[0].plot(df["time of day"], df[method], label=name, color=color)
axes[0].set_ylabel("Electrical Power [kW]")
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=6, fontsize=9)



variables = ["Space Heating [kW]", "DHW [kW]", "Nr. of available EV", "Outside Temperature [°C]", "Price [Cent/kWh]"]
for i, variable in enumerate(variables, start=1):
    axes[i].plot(df["time of day"], df[variable], color='blue')
    axes[i].set_ylabel(variable)

# for ax in axes:
ax = axes[-1]
ax.margins(x=0)
labels = ax.get_xticklabels()
ax.set_xticks([i for i, lbl in enumerate(labels) if
lbl.get_text().endswith('00')])
ax.tick_params(axis='x', rotation=90)



plt.xlabel("Time of Day")
plt.tight_layout()




combined_output_path = r"C:\Users\wi9632\Desktop\Results_End_MultiOpt\Load_Profile_Figure\Profiles_combined.png"
plt.savefig(combined_output_path, dpi=200)
plt.show()
