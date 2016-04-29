import pickle
import pandas as pd
import numpy as np
import copy
import sklearn.preprocessing
import sklearn.decomposition
from pprint import pprint
import xgboost as xgb

import machine_learning as ml


CLASSIFIERS = {"xgboost": xgb.XGBClassifier(n_estimators=50)}


def main():
    var_dict = extract_variable()
    df_dict, target_df, test_id_df = seprate_df_by_variable(var_dict)
    df_dict = PCA_analysis_per_variable(df_dict)
    df_all, target_all = PCA_analysis_all()
    print "PCA per varialbe"
    ml.learning_curve(CLASSIFIERS, df_dict['train'], target_df)
    print "PCA all"
    ml.learning_curve(CLASSIFIERS, df_all, target_all)


def PCA_analysis_per_variable(df_dict):
    new_df_dict = {"train": pd.DataFrame(), "test": pd.DataFrame()}
    for var_name, sub_df in df_dict.iteritems():
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(sub_df["train"])
        X_test = scaler.fit_transform(sub_df["test"])
        scaled_dict = {"train": X_train, "test": X_test}
        pca = sklearn.decomposition.PCA(n_components=0.999)
        pca.fit(X_train)
        accumulated_ratio = pca.explained_variance_ratio_.cumsum()
        for data_type in ["train", "test"]:
            X_new = pd.DataFrame(data=pca.transform(scaled_dict[data_type]))
            new_df_dict[data_type] = pd.concat([new_df_dict[data_type], X_new], axis="col")
            new_columns = range(len(new_df_dict[data_type].columns))
            new_df_dict[data_type].columns = new_columns
    return new_df_dict

def PCA_analysis_all():
    df = pd.read_csv("../../data/train_4-26.csv")
    df_target = df['TARGET']
    df.drop('TARGET', axis=1, inplace=True)
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(df)
    pca = sklearn.decomposition.PCA(n_components=0.99)
    return pd.DataFrame(data=pca.fit_transform(X)), df_target




def seprate_df_by_variable(var_dict):
    df = {"train": pd.read_csv("../../data/train_4-26.csv"),
          "test": pd.read_csv("../../data/test_4-26.csv")}
    df_dict = {}
    for key in var_dict.keys():
        df_dict[key] = {"train": pd.DataFrame(), "test": pd.DataFrame()}
    target_df = df["train"]['TARGET']
    test_id_df = df["test"]['ID']
    for var_name, column_names in var_dict.iteritems():
        for data_type in ["train", "test"]:
            df_dict[var_name][data_type] = df[data_type][column_names]
    return df_dict, target_df, test_id_df



def extract_variable():
    with open('../../data/column_names_pickle_dummps.txt', "r") as f:
        columns = pickle.loads(f.read())
    columns.remove("TARGET")
    var_dict = {}
    for column in columns:
        items = column.split("_")
        for item in items:
            if "var" in item:
                if "cte" in item:
                    item = item.replace("cte", "")
                if item in var_dict.keys():
                    var_dict[item].append(column)
                else:
                    var_dict[item] = [column]
    return var_dict

def dumps_column_names():
    df = pd.read_csv("../../data/train_4-26.csv")
    with open('../../data/column_names_pickle_dummps.txt', "w") as f:
        f.write(pickle.dumps(list(df.columns)))
        f.close()

if __name__ == '__main__':
    main()