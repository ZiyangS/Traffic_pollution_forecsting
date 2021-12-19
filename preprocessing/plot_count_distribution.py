import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# check distribution of vehicle across fuel consumption
data_list = os.listdir("./data")
print(data_list)
fuel_count_dict = {}
for data_file in data_list:
    data = pd.read_csv("./data/" + data_file)
    data = data.drop(['vehicle_type', 'fuel_type'], axis=1)
    # grouped same fuel consumption
    grouped_fuel_consumption_data = data.groupby("fuel_consumption").sum()
    grouped_fuel_consumption_data = grouped_fuel_consumption_data.reset_index()
    plotted_data = grouped_fuel_consumption_data.to_numpy()
    consumption = plotted_data[:,0]
    count = plotted_data[:,1]
    for pair in plotted_data:
        key = pair[0]
        value = pair[1]
        if key in fuel_count_dict.keys():
            fuel_count_dict[key] += value
        else:
            fuel_count_dict[key] = value

    # plot counts
    plt.bar(consumption, count)
    plt.xlabel('Fuel Consumption')
    plt.ylabel('Vehicle Count')
    plt.title('Distribution of Vehicles across fuel consumption' + data_file)
    # plt.show()
    plt.savefig("./count_plot/" + data_file[:4] + ".jpg")
    plt.clf()


total_consumption_count = [[key, fuel_count_dict[key]] for key in fuel_count_dict]
total_consumption_count = np.array(total_consumption_count)
print(total_consumption_count)
plt.bar(consumption, count)
plt.xlabel('Fuel Consumption')
plt.ylabel('Vehicle Count')
plt.title('Distribution of Vehicles across fuel consumption for all data')
plt.savefig("./count_plot/total_count.jpg")
plt.show()
