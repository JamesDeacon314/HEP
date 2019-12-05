import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
import pandas as pd

from stats import showStats
from model import modelfit
import load_data
gamma1_filename = load_data.gamma1_filename
neutron1_filename = load_data.neutron1_filename

data_gamma1 = []
data_neutron1 = []
x_vals = []
y_vals = []
z_vals = []
e_vals = []
count = 0
with open(gamma1_filename) as gamma1:
    gamma1_lines = gamma1.readlines()
    for row in gamma1_lines:
        entry = [float(x) for x in row.split()]
        if len(entry) == 4:
            x_vals.append(entry[0])
            y_vals.append(entry[1])
            z_vals.append(entry[3])
            e_vals.append(entry[2])
            count += 1
        else:
            data_gamma1.append(np.array([np.std(x_vals), np.std(y_vals), np.std(z_vals), np.std(e_vals), np.mean(x_vals), np.mean(y_vals), np.mean(z_vals), np.mean(e_vals), count, 1]))
            x_vals = []
            y_vals = []
            z_vals = []
            e_vals = []
            count = 0
with open(neutron1_filename) as neutron1:
    neutron1_lines = neutron1.readlines()
    for row in neutron1_lines:
        entry = [float(x) for x in row.split()]
        if len(entry) == 4:
            x_vals.append(entry[0])
            y_vals.append(entry[1])
            z_vals.append(entry[3])
            e_vals.append(entry[2])
            count += 1
        else:
            data_neutron1.append(np.array([np.std(x_vals), np.std(y_vals), np.std(z_vals), np.std(e_vals), np.mean(x_vals), np.mean(y_vals), np.mean(z_vals), np.mean(e_vals), count, 0]))
            x_vals = []
            y_vals = []
            z_vals = []
            e_vals = []
            count = 0

data = np.array(data_gamma1 + data_neutron1)

print("data shape: {}".format(data.shape))
print("some statistics...")
print("gamma data: ")
showStats(np.array(data_gamma1))
print("neutron data: ")
showStats(np.array(data_neutron1))

# data preprocessing
print("preprocessing data with standard scaler...")
scaler = preprocessing.StandardScaler().fit(data)
print("scaler mean: {}".format(scaler.mean_))
print("scaler scale: {}".format(scaler.scale_))
data = scaler.transform(data)
# showStats(data)

data_T = data.transpose()
x_coords = data_T[0]
y_coords = data_T[1]
z_coords = data_T[2]
E_coords = data_T[3]
xm_coords = data_T[4]
ym_coords = data_T[5]
zm_coords = data_T[6]
Em_coords = data_T[7]
count = data_T[8]
id_coords = data_T[9]

X = np.array([x_coords, y_coords, z_coords, E_coords, xm_coords, ym_coords, zm_coords, Em_coords, count]).transpose()
Y = np.round(id_coords, decimals=0)
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# create XGBoost model
model = xgb.XGBClassifier()
# grid = {'max_depth':10}
# model.set_params(**grid)
print(model)
model.fit(X_train, y_train)
# make predictions
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
print(np.array(predictions))
# evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: {}".format(accuracy))

alg = XGBClassifier(learning_rate=0.1, n_estimators=500, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.8, colsample_bytree=0.6,

                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

Xdf = pd.DataFrame(X_train, columns=['x_std', 'y_std', 'z_std', 'E_std', 'x_mean', 'y_mean', 'z_mean', 'E_mean', 'count'])
ydf = pd.DataFrame(y_train, columns=['id'])
ydf[ydf < 0] = 0
ydf[ydf > 0] = 1
modelfit(alg, Xdf, ydf)
