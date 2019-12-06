import numpy as np
import pandas as pd
import scipy

from sklearn import metrics

def model_report(y, y_pred, y_predprob, y_test, y_test_pred, y_test_predprob):
    print("\nModel Report")
    print("Accuracy (Train): %.4g" % metrics.accuracy_score(y.values, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

    print("Accuracy (Test): %.4g" % metrics.accuracy_score(y_test.values, y_test_pred))
    print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_test_predprob))
