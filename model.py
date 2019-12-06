import numpy as np
import pandas as pd
import scipy
import csv

import xgboost as xgb

from sklearn import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from model_report import model_report

def modelfit(alg, X, y, X_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=80):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X.values, label=y.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval = 500)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0])
        n_estimators_optimal = cvresult.shape[0]

    alg.fit(X, y,eval_metric='auc')

    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]

    dtest_predictions = alg.predict(X_test)
    dtest_predprob = alg.predict_proba(X_test)[:,1]

    model_report(y, dtrain_predictions, dtrain_predprob, y_test, dtest_predictions, dtest_predprob)

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

    with open('results1.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id,prediction'])
        for i in range(len(dtest_predictions)):
            writer.writerow([str(i) + ',' + str(dtest_predictions[i])])
