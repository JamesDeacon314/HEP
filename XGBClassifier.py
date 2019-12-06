import numpy as np
import pandas as pd
import scipy

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from model import modelfit
from preprocess import process_data
import load_data
gamma1_filename = load_data.gamma1_filename
neutron1_filename = load_data.neutron1_filename

(X_train, X_test, Y_train, Y_test) = process_data(gamma1_filename, neutron1_filename)

alg = XGBClassifier(learning_rate=0.1, n_estimators=500, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.8, colsample_bytree=0.6,

                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

Xdf = pd.DataFrame(X_train, columns=['x_std', 'y_std', 'z_std', 'E_std', 'x_mean', 'y_mean', 'z_mean', 'E_mean', 'count'])
Ydf = pd.DataFrame(Y_train, columns=['id'])
Xtdf = pd.DataFrame(X_test, columns=['x_std', 'y_std', 'z_std', 'E_std', 'x_mean', 'y_mean', 'z_mean', 'E_mean', 'count'])
Ytdf = pd.DataFrame(Y_test, columns=['id'])

modelfit(alg, Xdf, Ydf, Xtdf, Ytdf, "XGBC_results.csv", "XGBC_results_probs.csv")
