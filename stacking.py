import numpy as np
import pandas as pd
import scipy

from load_results import load_results
from save_results import save_results
from model_report import model_report_test
from preprocess import process_data
import load_data
gamma1_filename = load_data.gamma1_filename
neutron1_filename = load_data.neutron1_filename

def stack_results(Y, name1, name2, frac1, frac2, nameout1, nameout2):
    print("Loading first result...")
    res1 = load_results(name1, True)
    print("Loading second result...")
    res2 = load_results(name2, True)
    res1 = np.array([i * frac1 for i in res1])
    res2 = np.array([i * frac2 for i in res2])
    result = np.add(res1, res2)
    result_pred = np.round(result, decimals=0)
    model_report_test(Y, result_pred, result)
    save_results(nameout1, nameout2, result_pred, result)

(X_train, X_test, Y_train, Y_test) = process_data(gamma1_filename, neutron1_filename)
Y = pd.DataFrame(Y_test, columns=['id'])
stack_results(Y, "XGBC_results_probs.csv", "NN_results_probs.csv", 0.6, 0.35, "Stack_pred.csv", "Stack_predprob.csv")
