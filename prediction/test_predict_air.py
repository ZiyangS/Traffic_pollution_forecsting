import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
inventory_data = pd.read_excel("air quality prediction7.28 final version.xlsx", sheet_name = "variable",  index_colint=0)
inventory_data = inventory_data.drop(['Variables'], axis=1)
inventory_data = inventory_data.drop(['adt for cars and station wagons (km)', 'adt for vans, suvs, and trucks (km)'], axis=1)
inventory_array = inventory_data.to_numpy() # to numpy array, 18 x 30(28), each row is a year
inventory_array = np.delete(inventory_array, (8,17), axis=0) # 16 x 28
print(inventory_array.shape)
print(inventory_data.columns)

air_data = pd.read_excel("air quality prediction7.28 final version.xlsx", sheet_name = "air", index_colint=0)
air_data = air_data.drop(['Prediction'], axis=1)
air_array = air_data.to_numpy() # to numpy array, 18 x 10, each row is a year
air_array = np.delete(air_array, (8,17), axis=0) # 16 x 10
print(air_array.shape)


# variable_data = variable_data.drop(['four door number', 'SUV number', 'station wagon number', 'truck number',
#                                     'two door car number ', 'van number',], axis=1)
# variable_data = variable_data.drop(['Unnamed: 0'], axis=1)
# variable_name = list(variable_data.columns)
# variable_name[-2] = 'median total income'
# variable_name[-1] = 'employment rate'
# print(variable_name)
# plot correlation
# variable_data.columns = variable_name
# plt.figure(figsize=(12,10))
# cor = variable_data.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

# # delete less predictive variables
# variable_name = list(variable_data.columns)
# print(variable_name)
# variable_data = variable_data.drop(['proportion of males', 'proportion of females'], axis=1)
# variable_data = variable_data.drop(['station wagon weighted mpg'], axis=1)
# variable_data = variable_data.drop(['four door weighted mpg'], axis=1)
# variable_data = variable_data.drop(['truck weighted mpg'], axis=1)
# variable_data = variable_data.drop(['two door weighted mpg'], axis=1)
# variable_data = variable_data.drop(['van weighted mpg'], axis=1)
# variable_data = variable_data.drop(['median age'], axis=1)
# variable_data = variable_data.drop(['employment rate % (SK)'], axis=1)
# variable_data = variable_data.drop(['Unnamed: 0'], axis=1)
# variable_name = list(variable_data.columns)
# variable_name[-1] = 'median total income'
# print(variable_name)
# # variable_data = variable_data.drop(['average age'], axis=1)
# # variable_data = variable_data.drop(['persons median total income/dollars'], axis=1)
# # variable_data = variable_data.drop(['population all ages'], axis=1)

# # get data
# variable_array = variable_data.to_numpy() # to numpy array, 18 x 15, each row is a year
# print(variable_array.shape)
# # normalization or standardization, fit scaler on training data
# norm = StandardScaler().fit(variable_array)
# # transform training data
# variable_array = norm.transform(variable_array)
#
# train test split
X_train = inventory_array[0:15, :] # choose first 15 years as train
print(X_train.shape)
Y_train = air_array[0:15, :]# choose first 15 years as train
print(Y_train.shape)
X_test = inventory_array[15:, :]
Y_test = air_array[15:, :]
print(X_test.shape)
print(Y_test.shape)

# linear regression
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
Y_predicted = regr.predict(X_test)
print(Y_predicted)
print(Y_test)
# print(mean_absolute_error(Y_test, Y_predicted)) #
print(np.abs(Y_test - Y_predicted))
print(np.abs(Y_test - Y_predicted) / Y_test)

# LASSO
# regr = linear_model.Lasso(alpha=0.1)
# regr.fit(X_train, Y_train)
# Y_predicted = regr.predict(X_test)
# print(Y_predicted)
# print(Y_test)
# print(mean_absolute_error(Y_test, Y_predicted)) #

# decision tree
# clf = tree.DecisionTreeRegressor(min_samples_split=2)
# clf = clf.fit(X_train, Y_train)
# Y_predicted = clf.predict(X_test)
# print(Y_predicted)
# print(Y_test)
# print(mean_absolute_error(Y_test, Y_predicted)) #

# random forest
# regr = RandomForestRegressor()
# regr.fit(X_train, Y_train)
# Y_predicted = regr.predict(X_test)
# print(Y_predicted)
# print(Y_test)
# print(mean_absolute_error(Y_test, Y_predicted)) #

# from sklearn.ensemble import GradientBoostingRegressor
# Y_predicted = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X_train, Y_train).predict(X_test)
# print(Y_predicted)
# print(Y_test)
# print(mean_absolute_error(Y_test, Y_predicted)) #

import xgboost as xgb
# fitting
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', eval_metric="mape")).fit(X_train, Y_train)
Y_predicted = multioutputregressor.predict(X_test)
print(Y_predicted)
print(Y_test)
# print(mean_absolute_error(Y_test, Y_predicted)) #
print(np.abs(Y_test - Y_predicted))
print(np.abs(Y_test - Y_predicted) / Y_test)
num_estimator = len(multioutputregressor.estimators_)
total_feature_importance = np.zeros((28,))
for e in range(num_estimator):
    print(e)
    total_feature_importance += multioutputregressor.estimators_[e].feature_importances_
total_feature_importance = (total_feature_importance/10)
np.set_printoptions(precision=2)
print(total_feature_importance)

