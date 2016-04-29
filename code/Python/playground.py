import pandas as pd
import numpy as np
import pickle
import sklearn.preprocessing

df = pd.read_csv("../../data/train.csv")

print pd.concat([df, pd.DataFrame()], axis=1)
