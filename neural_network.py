import numpy as np
import pandas as pd
import scipy

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error

from model import modelfit
from preprocess3 import process_data
from model_report import model_report
from save_results import save_results
import load_data
gamma1_filename = load_data.gamma1_filename
neutron1_filename = load_data.neutron1_filename
gamma2_filename = load_data.gamma2_filename
neutron2_filename = load_data.neutron2_filename

(X_train, X_test, Y_train, Y_test) = process_data(gamma1_filename, gamma2_filename, neutron1_filename, neutron2_filename)

# Create model here given constraints in the problem
model = Sequential()

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.30))
model.add(Dense(600))
model.add(Activation('relu'))
model.add(Dropout(0.20))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dropout(0.20))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.10))
model.add(Dense(1))
#model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

fit = model.fit(X_train, Y_train, batch_size=170, nb_epoch=100)
model.summary()
train_predprob = model.predict(X_train)
test_predprob = model.predict(X_test)
train_pred = np.round(train_predprob, decimals=0)
test_pred = np.round(test_predprob, decimals=0)

train_predprob = train_predprob.transpose()[0]
test_predprob = test_predprob.transpose()[0]
train_pred = train_pred.transpose()[0]
test_pred = test_pred.transpose()[0]

Ydf = pd.DataFrame(Y_train, columns=['id'])
Ytdf = pd.DataFrame(Y_test, columns=['id'])

model_report(Ydf, train_pred, train_predprob, Ytdf, test_pred, test_predprob)
print(mean_squared_error(Ydf, train_predprob))
save_results("NN_train_results2.csv", "NN_train_probs2.csv", train_pred, train_predprob)
save_results("NN_results2.csv", "NN_results_probs2.csv", test_pred, test_predprob)
