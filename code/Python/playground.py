import pandas as pd
import numpy as np
import pickle

with open('../../data/column_names_pickle_dummps.txt', "r") as f:
    columns = pickle.loads(f.read())
print columns
