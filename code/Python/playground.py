import pandas as pd
import numpy as np
import pickle
import sklearn.preprocessing



df = pd.DataFrame(np.random.rand(4,4))

print df
print df.columns

new_names = np.array(range(len(df.columns))).cumsum()

df.columns = new_names
print df