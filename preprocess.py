import numpy as np
import pandas as pd
import scipy

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from stats import showStats

def process_data(gamma1_filename, neutron1_filename):
    data_gamma1 = []
    data_neutron1 = []
    x_vals = []
    y_vals = []
    z_vals = []
    e_vals = []
    count = 0
    with open(gamma1_filename) as gamma1:
        gamma1_lines = gamma1.readlines()
        for row in gamma1_lines:
            entry = [float(x) for x in row.split()]
            if len(entry) == 4:
                x_vals.append(entry[0])
                y_vals.append(entry[1])
                z_vals.append(entry[3])
                e_vals.append(entry[2])
                count += 1
            else:
                data_gamma1.append(np.array([np.std(x_vals), np.std(y_vals), np.std(z_vals), np.std(e_vals), np.mean(x_vals), np.mean(y_vals), np.mean(z_vals), np.mean(e_vals), count, 1]))
                x_vals = []
                y_vals = []
                z_vals = []
                e_vals = []
                count = 0
    with open(neutron1_filename) as neutron1:
        neutron1_lines = neutron1.readlines()
        for row in neutron1_lines:
            entry = [float(x) for x in row.split()]
            if len(entry) == 4:
                x_vals.append(entry[0])
                y_vals.append(entry[1])
                z_vals.append(entry[3])
                e_vals.append(entry[2])
                count += 1
            else:
                data_neutron1.append(np.array([np.std(x_vals), np.std(y_vals), np.std(z_vals), np.std(e_vals), np.mean(x_vals), np.mean(y_vals), np.mean(z_vals), np.mean(e_vals), count, 0]))
                x_vals = []
                y_vals = []
                z_vals = []
                e_vals = []
                count = 0

    data = np.array(data_gamma1 + data_neutron1)

    print("data shape: {}".format(data.shape))
    print("some statistics...")
    print("gamma data: ")
    showStats(np.array(data_gamma1))
    print("neutron data: ")
    showStats(np.array(data_neutron1))

    # data preprocessing
    print("preprocessing data with standard scaler...")
    scaler = preprocessing.StandardScaler().fit(data)
    print("scaler mean: {}".format(scaler.mean_))
    print("scaler scale: {}".format(scaler.scale_))

    data = scaler.transform(data).transpose()
    x_coords = data[0]
    y_coords = data[1]
    z_coords = data[2]
    E_coords = data[3]
    xm_coords = data[4]
    ym_coords = data[5]
    zm_coords = data[6]
    Em_coords = data[7]
    count = data[8]
    id_coords = data[9]

    X = np.array([x_coords, y_coords, z_coords, E_coords, xm_coords, ym_coords, zm_coords, Em_coords, count]).transpose()
    Y = np.round(id_coords, decimals=0)
    Y[Y < 0] = 0

    seed = 7
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return (X_train, X_test, Y_train, Y_test)
