import numpy as np
import pandas as pd
import scipy

from preprocess3 import process_data
import load_data

from sklearn import svm

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from model_report import model_report
from save_results import save_results

def modelfit(alg, X, y, X_test, y_test, name1, name2, name3, name4):

    alg.fit(X, y)

    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]

    dtest_predictions = alg.predict(X_test)
    dtest_predprob = alg.predict_proba(X_test)[:,1]

    model_report(y, dtrain_predictions, dtrain_predprob, y_test, dtest_predictions, dtest_predprob)
    print(mean_squared_error(y_test, dtest_predprob))
    save_results(name1, name2, dtest_predictions, dtest_predprob)


gamma1_filename = load_data.gamma1_filename
neutron1_filename = load_data.neutron1_filename
gamma2_filename = load_data.gamma2_filename
neutron2_filename = load_data.neutron2_filename

(X_train, X_test, Y_train, Y_test) = process_data(gamma1_filename, gamma2_filename, neutron1_filename, neutron2_filename)

alg = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
			tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1,
			 decision_function_shape='ovr')

# Xdf = pd.DataFrame(X_train, columns=['x_std', 'y_std', 'z_std', 'E_std', 'x_mean', 'y_mean', 'z_mean', 'E_mean', 'w_x', 'w_y', 'w_z', 'count'])
Xdf = pd.DataFrame(X_train, columns=['x_std', 'y_std', 'z_std', 'E_std', 'x_mean', 'y_mean', 'z_mean', 'E_mean', 'w_x', 'w_y', 'w_z', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'emin', 'emax', 'count'])
Ydf = pd.DataFrame(Y_train, columns=['id'])
Xtdf = pd.DataFrame(X_test, columns=['x_std', 'y_std', 'z_std', 'E_std', 'x_mean', 'y_mean', 'z_mean', 'E_mean', 'w_x', 'w_y', 'w_z', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'emin', 'emax', 'count'])
Ytdf = pd.DataFrame(Y_test, columns=['id'])

modelfit(alg, Xdf, Ydf, Xtdf, Ytdf, "SVM_results.csv", "SVM_results_probs.csv", "SVM_train_results.csv", "SVM_trains_probs.csv")
