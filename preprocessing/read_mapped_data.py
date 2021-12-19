import os
import pandas as pd
import numpy as np



# if __name__ == '__main__':
#     print_hi('PyCharm')


# check value of data
# data_list = os.listdir("./data")
# print(data_list)
# for data_file in data_list:
#     data = pd.read_csv("./data/" + data_file)
    # print(len(np.unique(data.fuel_consumption))) #from 2001 -2018, data owns 109-116 different values
    # print(np.unique(data.vehicle_type)) # ['SUV' 'four_door_car' 'station_wagon' 'truck' 'two_door_car' 'van']
    # print(len(np.unique(data.vehicle_type))) # only 6 differnt values
    # print(np.unique(data.fuel_type)) # ['diesel' 'gasoline']
    # print(len(np.unique(data.fuel_type))) # only 2 differnt values


# create mapping dict which replace categorical data with oridinal values
data = pd.read_csv("./data/2001.csv")
# mapping vehicle types to ordinal values, as we know there are only 6 different vehicle types
unique_vehicle_type = np.unique(data.vehicle_type)
mapped_vechicle_value = list(range(len(unique_vehicle_type)))
vechicle_value_mapping = dict(zip(unique_vehicle_type, mapped_vechicle_value))
# print(vechicle_value_mapping)
# mapping fuel type sto ordinal values, as we know there are only 2 different fuel types
unique_fuel_type = np.unique(data.fuel_type)
mapped_fuel_value = list(range(len(unique_fuel_type)))
fuel_type_mapping = dict(zip(unique_fuel_type, mapped_fuel_value))
# print(fuel_type_mapping)


# replace categorical data with oridinal values for each yearly data
data_list = os.listdir("./data")
mapping_data_dict = {} # store mapped data
for data_file in data_list:
    yearly_data = pd.read_csv("./data/" + data_file)
    yearly_data = yearly_data.replace({"vehicle_type": vechicle_value_mapping})
    yearly_data = yearly_data.replace({"fuel_type": fuel_type_mapping})
    mapping_data_dict[data_file] = yearly_data
    mapping_data_dict[data_file].to_csv("./mapped_data/" + data_file, index=False) # save processed data
print(mapping_data_dict)
