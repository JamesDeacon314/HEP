import numpy as np
import pandas as pd
import scipy

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import modelfit
from preprocess import process_data
import load_data
gamma1_filename = load_data.gamma1_filename
neutron1_filename = load_data.neutron1_filename

data = process_data(gamma1_filename, neutron1_filename)

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
Y[Y < 0] = 0

seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

alg = XGBClassifier(learning_rate=0.1, n_estimators=500, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.8, colsample_bytree=0.6,

                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

Xdf = pd.DataFrame(X_train, columns=['x_std', 'y_std', 'z_std', 'E_std', 'x_mean', 'y_mean', 'z_mean', 'E_mean', 'count'])
ydf = pd.DataFrame(y_train, columns=['id'])
Xtdf = pd.DataFrame(X_test, columns=['x_std', 'y_std', 'z_std', 'E_std', 'x_mean', 'y_mean', 'z_mean', 'E_mean', 'count'])
ytdf = pd.DataFrame(y_test, columns=['id'])
modelfit(alg, Xdf, ydf, Xtdf, ytdf)
