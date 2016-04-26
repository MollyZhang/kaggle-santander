import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb
import matplotlib.pyplot as plt


def main():
    train = pd.read_csv("../../data/train.csv")
    test = pd.read_csv("../../data/test.csv")
    print train.describe().shape
    train, test = remove_0_variant_features(train, test)
    print train.describe().shape
    train, test = remove_duplicate_features(train, test)
    print train.describe().shape

def remove_duplicate_features(train, test):
    remove = []
    cols = train.columns
    for i in range(len(cols)-1):
        v = train[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v,train[cols[j]].values):
                remove.append(cols[j])
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    return train, test





def remove_0_variant_features(train, test):
    # remove ID and features with no variance
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    return train, test


if __name__ == "__main__":
    main()
