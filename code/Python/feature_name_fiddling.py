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
    pass




def experiment_4_26():
    ### seperate all variables into those containing var1, var2, ..var 46
    ### do seperate PCA in each variable group, comparing with PCA for all
    ### conclusion: confirmed linkage within variable names, however, PCA is not the answer


    df = {"train": pd.read_csv("../../data/train_4-26.csv"),
          "test": pd.read_csv("../../data/test_4-26.csv")}
    label = df["train"]["TARGET"]
    data = df["train"].drop("TARGET", axis=1)
    var_dict = extract_variable()
    df_dict, target_df, test_id_df = seprate_df_by_variable(var_dict, df)
    df_dict = PCA_analysis_per_variable(df_dict)
    df_all, target_all = PCA_analysis_all()
    print "no PCA"
    print ml.nfold_cross_validation(data, label, CLASSIFIERS)
    print "PCA per varialbe"
    print ml.nfold_cross_validation(df_dict['train'], target_df, CLASSIFIERS)
    print "PCA all"
    print ml.nfold_cross_validation(df_all, target_all, CLASSIFIERS)





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
    pca = sklearn.decomposition.PCA(n_components=0.999)
    return pd.DataFrame(data=pca.fit_transform(X)), df_target




def seprate_df_by_variable(var_dict, df):
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