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

new_variable_name = ['FDFC',
       'SUVFC',
       'SWFC',
       'TFC',
       'TDFC',
       'VFC', 'FDRQ',
       'SUVRQ', 'SWRQ', 'TRQ',
       'TDRQ', 'VRQ', 'POP',
       'POM', 'POF',
       'POC', 'AGE',
       'EMPL', 'INCO',
       'GP', 'DP',
       'DRQ', 'GRQ',
       'PRQ', 'FFRQ',
       'NGFQ', 'EFQ',
       'EHGFQ']
inventory_data.columns = new_variable_name

plt.figure(figsize=(16, 10))
heatmap = sns.heatmap(inventory_data.corr(), cmap="Blues", vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':10}, pad=12)
# plt.xticks(rotation=30)
plt.show()

