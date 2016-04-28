import pandas as pd
import numpy as np

x = pd.DataFrame(data=range(20), columns = ["hey"])


x['hey'] = np.where((x['hey'] > 10) & (x['hey'] < 15), 1, x['hey'])
print x



