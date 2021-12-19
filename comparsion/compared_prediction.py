import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
inventory_data = pd.read_excel("2001-2018 inventory.xlsx", sheet_name = "vehicle type", index_colint=0)
inventory_data = inventory_data.drop(labels=range(18, 44), axis=0)
inventory_data = inventory_data.drop(labels=[2,5,8,11,14,17], axis=0)
inventory_data = inventory_data.drop(['Unnamed: 19', 'Unnamed: 20'], axis=1)
inventory_data = inventory_data.drop(['Unnamed: 0'], axis=1)
inventory_array = inventory_data.to_numpy() # to numpy array, 12 x 18, each column is a year
inventory_array = np.transpose(inventory_array) # reshape, 18 x 12, each row is a year
print(inventory_array)


variable_data = pd.read_excel("2001-2018 inventory.xlsx", sheet_name = "2001-2018", index_colint=0)
variable_data = variable_data.drop(labels=range(18, 42), axis=0)
variable_data = variable_data.drop(['land use diversity', 'bus stop density', 'road density'], axis=1)
variable_data = variable_data.drop(['Unnamed: 0'], axis=1)
variable_array = variable_data.to_numpy() # to numpy array, 18 x 21, each row is a year
# print(variable_array.shape)

X_train = variable_array[0:17, :] # choose first 17 years as train
print(X_train.shape)
Y_train = inventory_array[0:17, :]# choose first 17  years as train
print(Y_train.shape)
X_test = variable_array[17:18, :]
Y_test = inventory_array[17:18, :]
print(X_test.shape)
print(Y_test.shape)



# comparison methods
# linear regression
# regr = linear_model.LinearRegression() # Do not use fit_intercept = False if you have removed 1 column after dummy encoding
# regr.fit(X_train, Y_train)
# Y_predicted = regr.predict(X_test)
# print(mean_absolute_error(Y_test, Y_predicted))

# LASSO
# regr = linear_model.Lasso(alpha=0.1)
# regr.fit(X_train, Y_train)
# Y_predicted = regr.predict(X_test)
# print(mean_absolute_error(Y_test, Y_predicted))

# decision tree
# clf = tree.DecisionTreeRegressor(min_samples_split=2)
# clf = clf.fit(X_train, Y_train)
# Y_predicted = clf.predict(X_test)
# print(mean_absolute_error(Y_test, Y_predicted))

# random forest
# regr = RandomForestRegressor()
# regr.fit(X_train, Y_train)
# Y_predicted = regr.predict(X_test)
# print(mean_absolute_error(Y_test, Y_predicted)) # 1516.1054166666665

# multioutput + ridge regression
# clf = MultiOutputRegressor(linear_model.Ridge(random_state=123)).fit(X_train, Y_train)
# Y_predicted = clf.predict(X_test)
# print(mean_absolute_error(Y_test, Y_predicted)) #4604

# multioutput + random forest
# max_depth = 30
# regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
#                                                           max_depth=max_depth,
#                                                           random_state=0))
# regr_multirf.fit(X_train, Y_train)
# regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth,
#                                 random_state=2)
# regr_rf.fit(X_train, Y_train)
# y_multirf = regr_multirf.predict(X_test)
# y_rf = regr_rf.predict(X_test)
# print(mean_absolute_error(Y_test, y_multirf)) # 1265
# print(mean_absolute_error(Y_test, y_rf)) # 1131

# gradient boosting
# from sklearn.ensemble import GradientBoostingRegressor
# Y_predicted = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X_train, Y_train).predict(X_test)
# print(mean_absolute_error(Y_test, Y_predicted)) # 896

# extra tree
# from sklearn.tree import ExtraTreeRegressor
# extra_tree = ExtraTreeRegressor(random_state=0)
# reg = BaggingRegressor(extra_tree, random_state=0).fit(X_train, Y_train)
# Y_predicted = chain.predict(X_test)
# print(mean_absolute_error(Y_test, Y_predicted)) # 951.3333333333334


