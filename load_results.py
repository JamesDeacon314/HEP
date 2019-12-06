import numpy as np
import pandas as pd
import scipy
import csv

def load_results(name, isHeader):
    results = []
    with open(name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if isHeader:
                isHeader = False
            else:
                for value in row:
                    results.append(float(value))
    return results
