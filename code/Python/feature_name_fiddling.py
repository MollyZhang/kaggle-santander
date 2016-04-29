import pickle
import pandas as pd
import numpy as np
import copy
import sklearn.preprocessing
import sklearn.decomposition

from pprint import pprint



def main():
    var_dict = extract_variable()
    df_dict, target_df, test_id_df = seprate_df_by_variable(var_dict)
    df_dict = PCA_analysis(df_dict)
    for df in df_dict.itervalues():
        print df.shape



def PCA_analysis(df_dict):
    new_df_dict = {"train": pd.DataFrame(), "test": pd.DataFrame()}
    for data_type in ["train", "test"]:
        for var_name, sub_df in df_dict[data_type].iteritems():
            scaler = sklearn.preprocessing.StandardScaler()
            X = scaler.fit_transform(sub_df)
            num_features = sub_df.shape[1]
            pca = sklearn.decomposition.PCA(n_components=0.999)
            X_new = pd.DataFrame(data=pca.fit_transform(X))
            accumulated_ratio = pca.explained_variance_ratio_.cumsum()
            new_df_dict[data_type] = pd.concat([new_df_dict[data_type], X_new], axis="col")
            print len(new_df_dict[data_type].columns)

    return new_df_dict



def seprate_df_by_variable(var_dict):
    df = {"train": pd.read_csv("../../data/train_4-26.csv"),
          "test": pd.read_csv("../../data/test_4-26.csv")}
    df_dict = {"train": {}, "test": {}}
    target_df = df["train"]['TARGET']
    test_id_df = df["test"]['ID']
    for data_type in ["train", "test"]:
        for var_name, column_names in var_dict.iteritems():
            df_dict[data_type][var_name] = df[data_type][column_names]
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