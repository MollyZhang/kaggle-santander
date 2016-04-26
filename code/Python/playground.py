import pandas as pd
import numpy as np

train = pd.read_csv("../../data/train.csv")
print train['TARGET'].as_matrix()