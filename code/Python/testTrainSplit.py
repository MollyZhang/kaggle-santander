import pandas as pd
from sklearn.cross_validation import train_test_split

df = pd.read_csv("../../data/trainReduced.csv")

# Remove column "Unnamed: 0" (artifact of data write in R)
df = df.drop(df.columns[[0]], axis = 1)

# Replace IDs with int vals
df.ix[:, 0] = range(1, len(df) + 1)


# Spit into test/train----------------------------

# First split out TARGET column, to confrom to sklearn style
y = df.ix[:, 'TARGET']
X = df.drop('TARGET', axis = 1)

# Convert TARGET back to 0, 1 vals:
y[y < 0] = 0
y[y > 0] = 1


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 1, stratify = y)
