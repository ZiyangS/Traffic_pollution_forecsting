import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
# from sklearn.metrics import mean_absolute_percentage_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
inventory_data = pd.read_excel("air quality prediction7.28 final version.xlsx", sheet_name = "variable",  index_colint=0)
inventory_data = inventory_data.drop(['Variables'], axis=1)
inventory_data = inventory_data.drop(['adt for cars and station wagons (km)', 'adt for vans, suvs, and trucks (km)'], axis=1)
inventory_array = inventory_data.to_numpy() # to numpy array, 18 x 30(28), each row is a year
# inventory_array = np.delete(inventory_array, (8,17), axis=0) # 16 x 28
print(inventory_array.shape)

air_data = pd.read_excel("air quality prediction7.28 final version.xlsx", sheet_name = "air", index_colint=0)
air_data = air_data.drop(['Prediction'], axis=1)
air_array = air_data.to_numpy() # to numpy array, 18 x 10, each row is a year
air_array = np.delete(air_array, (8,17), axis=0) # 16 x 10
print(air_array.shape)

# co_H09 (ppm)
X_train = inventory_array[0:16, :] # choose first 15 years as train
Y_train = air_array[0:16, 0]# choose first 15 years as train
X_test = inventory_array[16:, :]
Y_test = air_array[16:, 0]
regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric="mape").fit(X_train, Y_train)
Y_predicted = regressor.predict(X_test)
print(Y_predicted)
print(Y_test)
print(np.abs(Y_test - Y_predicted))
print(np.mean(np.abs(Y_test - Y_predicted) / Y_test))

# co_H18 (ppm)
X_train = inventory_array[0:16, :] # choose first 15 years as train
Y_train = air_array[0:16, 1]# choose first 15 years as train
X_test = inventory_array[16:, :]
Y_test = air_array[16:, 1]
regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric="mape").fit(X_train, Y_train)
Y_predicted = regressor.predict(X_test)
print(Y_predicted)
print(Y_test)
print(np.abs(Y_test - Y_predicted))
print(np.mean(np.abs(Y_test - Y_predicted) / Y_test))

# NOx_H09 (ppb)
X_train = inventory_array[0:15, :] # choose first 15 years as train
Y_train = air_array[0:15, 2]# choose first 15 years as train
X_test = inventory_array[15:, :]
Y_test = air_array[15:, 2]
Y_predicted = xgb.XGBRegressor(random_state=0).fit(X_train, Y_train).predict(X_test)
print(Y_predicted)
print(Y_test)
print(np.abs(Y_test - Y_predicted))
print(np.mean(np.abs(Y_test - Y_predicted) / Y_test))

# NOx_H18 (ppb)
X_train = inventory_array[0:17, :] # choose first 15 years as train
Y_train = air_array[0:17, 3]# choose first 15 years as train
X_test = inventory_array[17:, :]
Y_test = air_array[17:, 3]
Y_predicted = GradientBoostingRegressor(random_state=0).fit(X_train, Y_train).predict(X_test)
print(Y_predicted)
print(Y_test)
print(np.abs(Y_test - Y_predicted))
print(np.mean(np.abs(Y_test - Y_predicted) / Y_test))

# pm2.5_H09 ( ug/m3)
X_train = inventory_array[0:17, :] # choose first 15 years as train
Y_train = air_array[0:17, 4]# choose first 15 years as train
X_test = inventory_array[17:, :]
Y_test = air_array[17:, 4]
Y_predicted = xgb.XGBRegressor(n_estimators=300, max_depth=4, random_state=0).fit(X_train, Y_train).predict(X_test)
print(Y_predicted)
print(Y_test)
print(np.abs(Y_test - Y_predicted))
print((np.abs(Y_test - Y_predicted) / Y_test))
print(np.mean(np.abs(Y_test - Y_predicted) / Y_test))

# pm2.5_H18 ( ug/m3)
X_train = inventory_array[0:16, :] # choose first 15 years as train
Y_train = air_array[0:16, 5]# choose first 15 years as train
X_test = inventory_array[16:, :]
Y_test = air_array[16:, 5]
regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric="mape").fit(X_train, Y_train)
Y_predicted = regressor.predict(X_test)
print(Y_test)
print(np.abs(Y_test - Y_predicted))
print((np.abs(Y_test - Y_predicted) / Y_test))
print(np.mean(np.abs(Y_test - Y_predicted) / Y_test))


# so2_H09 ( ppb)
X_train = inventory_array[0:17, :] # choose first 15 years as train
Y_train = air_array[0:17, 8]# choose first 15 years as train
X_test = inventory_array[17:, :]
Y_test = air_array[17:, 8]
regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric="mape").fit(X_train, Y_train)
Y_predicted = regressor.predict(X_test)
print(Y_test)
print(np.abs(Y_test - Y_predicted))
print((np.abs(Y_test - Y_predicted) / Y_test))
print(np.mean(np.abs(Y_test - Y_predicted) / Y_test))


# so2_H18 ( ppb)
X_train = inventory_array[0:17, :] # choose first 15 years as train
Y_train = air_array[0:17, 9]# choose first 15 years as train
X_test = inventory_array[17:, :]
Y_test = air_array[17:, 9]
regressor = xgb.XGBRegressor(objective='reg:squarederror', eval_metric="mape").fit(X_train, Y_train)
Y_predicted = regressor.predict(X_test)
print(Y_test)
print(Y_predicted)
print(np.abs(Y_test - Y_predicted))
print((np.abs(Y_test - Y_predicted) / Y_test))
print(np.mean(np.abs(Y_test - Y_predicted) / Y_test))
