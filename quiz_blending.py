import numpy as np
import pandas as pd

from load_results import load_results

x_squared = 0.5
# results = ['TIMESVD4_(7.08%).txt', 'TIMESVD6_50factors_(6.78%).txt']

# all
result_files = ['XGBC_results_probs.csv', 'NN_results_probs2.csv', 'SVM_results_probs.csv']#, 'bc_results_probs.csv']#,
#'rfc_results_probs.csv', 'logreg_results_probs.csv', 'dt_results_probs.csv']

#result_files = ['XGBC_results_probs.csv', 'NN_results_probs2.csv']

# all

RMSE = [0.006731, 0.008036, 0.011048]#, 0.013775]#, 0.017775, 0.021000, 0.02875]

#RMSE = [0.010363, 0.011491]

A = []
for i, result_file in enumerate(result_files):
    results = load_results(result_file, True)
    if A != []:
        A = np.column_stack((A, results))
    else:
        A = results

sum_x_squared = x_squared * len(A[:,0])
RMSE_constant = [i * i * len(A[:,0]) for i in RMSE]
print(sum_x_squared)
print(RMSE_constant)
result = np.matmul(np.linalg.inv(np.matmul(A.T, A)), [0.5 * (sum(j*j for j in
A[:,i]) + sum_x_squared - RMSE_constant[i]) for i in range(len(A.T))])

# result[5] = result[5] - .8
# result = np.array(result) * 1.8
result = result / sum(result)
print(result)
result = (1 - result)
result = result / sum(result)
print(result)
blended_results = sum([np.array(A[:,i]) * result[i] for i in range(len(A.T))])
from save_results import save_prob_results
save_prob_results("belnded_results.csv", blended_results)


from preprocess3 import process_data
import load_data
gamma1_filename = load_data.gamma1_filename
neutron1_filename = load_data.neutron1_filename
gamma2_filename = load_data.gamma2_filename
neutron2_filename = load_data.neutron2_filename

(X_train, X_test, Y_train, Y_test) = process_data(gamma1_filename, gamma2_filename, neutron1_filename, neutron2_filename)
Ytdf = pd.DataFrame(Y_test, columns=['id'])
from model_report import model_report_test
model_report_test(Ytdf, np.round(blended_results, decimals=0), blended_results)
