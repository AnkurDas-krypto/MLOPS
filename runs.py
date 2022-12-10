#this file is created to run the ml model multiple times

import numpy as np
import os

alphas = np.linspace(0.1, 1.0, 5)
l1_ratio = np.linspace(0.1, 1.0, 5)

for alpha in alphas:
    for ratio in l1_ratio:
        os.system(f"simple_ml_model.py -a {alpha} -l1 {ratio}")