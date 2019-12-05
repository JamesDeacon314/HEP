import numpy as np
import pandas as pd
import scipy

def showStats(data):
  data_T = data.transpose()
  x_coords = data_T[0]
  y_coords = data_T[1]
  z_coords = data_T[3]
  E_coords = data_T[2]
  print("x mean: {}".format(np.mean(x_coords)))
  print("x std: {}".format(np.std(x_coords)))

  print("y mean: {}".format(np.mean(y_coords)))
  print("y std: {}".format(np.std(y_coords)))

  print("z mean: {}".format(np.mean(z_coords)))
  print("z std: {}".format(np.std(z_coords)))

  print("E mean: {}".format(np.mean(E_coords)))
  print("E std: {}".format(np.std(E_coords)))
