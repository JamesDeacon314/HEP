import numpy as np
import pandas as pd
import scipy

from load_results import load_results
from save_results import save_results
from model_report import model_report_test
from model_report import model_report_get
from preprocess2 import process_data
import load_data
gamma1_filename = load_data.gamma1_filename
neutron1_filename = load_data.neutron1_filename
gamma2_filename = load_data.gamma2_filename
neutron2_filename = load_data.neutron2_filename

def stack_results_2(Y, Yt, name1, name2, name3, name4, nameout1, nameout2):
    print("Loading first result...")
    res1 = load_results(name1, True)
    print("Loading second result...")
    res2 = load_results(name2, True)
    max_acc = 0
    maxx_auc = 0;
    max_frac = 0;
    for j in range(101):
        frac = j/100.0
        res_1 = np.array([i * frac for i in res1])
        res_2 = np.array([i * (1 - frac) for i in res2])
        result = np.add(res_1, res_2)
        result_pred = np.round(result, decimals=0)
        (acc, auc) = model_report_get(Y, result_pred, result)
        if (acc > max_acc):
            max_acc = acc
            max_auc = auc
            max_frac = frac
    print("The max fraction is: %f" % max_frac)
    print("Loading first result...")
    res1 = load_results(name3, True)
    print("Loading second result...")
    res2 = load_results(name4, True)
    res_1 = np.array([i * max_frac for i in res1])
    res_2 = np.array([i * (1 - max_frac) for i in res2])
    result = np.add(res_1, res_2)
    result_pred = np.round(result, decimals=0)
    model_report_test(Yt, result_pred, result)
    save_results(nameout1, nameout2, result_pred, result)

(X_train, X_test, Y_train, Y_test) = process_data(gamma1_filename, gamma2_filename, neutron1_filename, neutron2_filename)
Y = pd.DataFrame(Y_train, columns=['id'])
Yt = pd.DataFrame(Y_test, columns=['id'])
stack_results_2(Y, Yt, "XGBC_trains_probs.csv", "NN_train_probs.csv", "XGBC_results_probs.csv", "NN_results_probs.csv", "Stack_pred.csv", "Stack_predprob.csv")

def stack_results_3(Y, name1, name2, name3, nameout1, nameout2):
    print("Loading first result...")
    res1 = load_results(name1, True)
    print("Loading second result...")
    res2 = load_results(name2, True)
    print("Loading third result...")
    res3 = load_results(name3, True)
    max_acc = 0
    maxx_auc = 0;
    max_frac1 = 0;
    max_frac2 = 0;
    for j in range(101):
        frac1 = j/100.0
        for k in range(101-j):
            frac2 = k/100.0
            res_1 = np.array([i * frac1 for i in res1])
            res_2 = np.array([i * frac2 for i in res2])
            res_3 = np.array([i * (1 - frac1 - frac2) for i in res3])
            result = np.add(res_1, res_2, res_3)
            result_pred = np.round(result, decimals=0)
            (acc, auc) = model_report_get(Y, result_pred, result)
            if (acc > max_acc):
                max_acc = acc
                max_auc = auc
                max_frac1 = frac1
                max_frac2 = frac2
    print("The max fractions are: %f, %f" % (max_frac1, max_frac2))
    res_1 = np.array([i * max_frac1 for i in res1])
    res_2 = np.array([i * max_frac2 for i in res2])
    res_3 = np.array([i * (1 - max_frac1 - max_frac2) for i in res3])
    result = np.add(res_1, res_2, res_3)
    result_pred = np.round(result, decimals=0)
    model_report_test(Y, result_pred, result)
    save_results(nameout1, nameout2, result_pred, result)

#(X_train, X_test, Y_train, Y_test) = process_data(gamma1_filename, neutron1_filename)
#Y = pd.DataFrame(Y_test, columns=['id'])
#stack_results_3(Y, "XGBC_results_probs.csv", "NN_results_probs.csv", "rfc_results_probs.csv", "Stack_pred.csv", "Stack_predprob.csv")
