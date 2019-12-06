import numpy as np
import pandas as pd
import scipy
import csv

def save_results(name1, name2, results, predprobs):
    with open(name1, 'w') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(['prediction'])
        writer.writerow(results)

    with open(name2, 'w') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(['prediction'])
        writer.writerow(predprobs)
