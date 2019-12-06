import numpy as np
import pandas as pd
import scipy

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import GridSearchCV

from model import modelfit
from preprocess import process_data
import load_data
gamma1_filename = load_data.gamma1_filename
neutron1_filename = load_data.neutron1_filename

(X_train, X_test, Y_train, Y_test) = process_data(gamma1_filename, neutron1_filename)

Xdf = pd.DataFrame(X_train, columns=['x_std', 'y_std', 'z_std', 'E_std', 'x_mean', 'y_mean', 'z_mean', 'E_mean', 'count'])
Ydf = pd.DataFrame(Y_train, columns=['id'])

# 5, 0.1, 0.7, 0.5, 7
params = {'min_child_weight':[4,5,6], 'gamma':[i/20.0 for i in range(1,3)],  'subsample':[i/10.0 for i in range(6,8)],
'colsample_bytree':[i/10.0 for i in range(4,6)], 'max_depth': [6,7,8]}

# Initialize XGB and GridSearch
xgb = XGBClassifier(nthread=-1)

print("BEGIN THE GRIDSEARCH")
grid = GridSearchCV(xgb, params)
grid.fit(Xdf, Ydf)

# Print the r2 score
print(grid.cv_results_)
print(grid.best_params_)
print(grid.best_score_)
