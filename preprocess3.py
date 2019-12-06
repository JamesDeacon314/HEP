import numpy as np
import pandas as pd
import scipy
import math

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from stats import showStats

def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (math.sqrt(variance))

def process_data(gamma1_filename, gamma2_filename, neutron1_filename, neutron2_filename):
    data_gamma1 = []
    data_gamma2 = []
    data_neutron1 = []
    data_neutron2 = []
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
                data_gamma1.append(np.array([np.std(x_vals), np.std(y_vals), np.std(z_vals), np.std(e_vals), np.mean(x_vals), np.mean(y_vals), np.mean(z_vals), np.mean(e_vals), weighted_std(x_vals, e_vals), weighted_std(y_vals, e_vals), weighted_std(z_vals, e_vals), np.amin(x_vals), np.amax(x_vals), np.amin(y_vals), np.amax(y_vals), np.amin(z_vals), np.amax(z_vals), np.amin(e_vals), np.amax(e_vals), count, 1]))

                x_vals = []
                y_vals = []
                z_vals = []
                e_vals = []
                count = 0
    with open(gamma2_filename) as gamma2:
        gamma2_lines = gamma2.readlines()
        for row in gamma2_lines:
            entry = [float(x) for x in row.split()]
            if len(entry) == 4:
                x_vals.append(entry[0])
                y_vals.append(entry[1])
                z_vals.append(entry[3])
                e_vals.append(entry[2])

                count += 1
            else:
                data_gamma2.append(np.array([np.std(x_vals), np.std(y_vals), np.std(z_vals), np.std(e_vals), np.mean(x_vals), np.mean(y_vals), np.mean(z_vals), np.mean(e_vals), weighted_std(x_vals, e_vals), weighted_std(y_vals, e_vals), weighted_std(z_vals, e_vals), np.amin(x_vals), np.amax(x_vals), np.amin(y_vals), np.amax(y_vals), np.amin(z_vals), np.amax(z_vals), np.amin(e_vals), np.amax(e_vals), count, 1]))

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
                data_neutron1.append(np.array([np.std(x_vals), np.std(y_vals), np.std(z_vals), np.std(e_vals), np.mean(x_vals), np.mean(y_vals), np.mean(z_vals), np.mean(e_vals), weighted_std(x_vals, e_vals), weighted_std(y_vals, e_vals), weighted_std(z_vals, e_vals), np.amin(x_vals), np.amax(x_vals), np.amin(y_vals), np.amax(y_vals), np.amin(z_vals), np.amax(z_vals), np.amin(e_vals), np.amax(e_vals), count, 0]))
                x_vals = []
                y_vals = []
                z_vals = []
                e_vals = []
                count = 0
    with open(neutron2_filename) as neutron2:
        neutron2_lines = neutron2.readlines()
        for row in neutron2_lines:
            entry = [float(x) for x in row.split()]
            if len(entry) == 4:
                x_vals.append(entry[0])
                y_vals.append(entry[1])
                z_vals.append(entry[3])
                e_vals.append(entry[2])
                count += 1
            else:
                data_neutron2.append(np.array([np.std(x_vals), np.std(y_vals), np.std(z_vals), np.std(e_vals), np.mean(x_vals), np.mean(y_vals), np.mean(z_vals), np.mean(e_vals), weighted_std(x_vals, e_vals), weighted_std(y_vals, e_vals), weighted_std(z_vals, e_vals), np.amin(x_vals), np.amax(x_vals), np.amin(y_vals), np.amax(y_vals), np.amin(z_vals), np.amax(z_vals), np.amin(e_vals), np.amax(e_vals), count, 0]))
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
    w_xcoords = data[8]
    w_ycoords = data[9]
    w_zcoords = data[10]
    x_min = data[11]
    x_max = data[12]
    y_min = data[13]
    y_max = data[14]
    z_min = data[15]
    z_max = data[16]
    e_min = data[17]
    e_max = data[18]
    count = data[19]
    id_coords = data[20]

    X = np.array([x_coords, y_coords, z_coords, E_coords, xm_coords, ym_coords, zm_coords, Em_coords, w_xcoords, w_ycoords, w_zcoords, x_min, x_max, y_min, y_max, z_min, z_max, e_min, e_max, count]).transpose()
    Y = np.round(id_coords, decimals=0)
    Y[Y < 0] = 0

    seed = 7
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return (X_train, X_test, Y_train, Y_test)
