import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def main():
    train = pd.read_csv("../../data/train_4-26.csv")
    test = pd.read_csv("../../data/test_4-26.csv")
    train, test = feature_engineering(train, test)




    # train.to_csv('../../data/train_4-26.csv', index=False)
    # test.to_csv('../../data/test_4-26.csv', index=False)

def feature_engineering(train, test):
    feature_names = ['delta_imp_aport_var13_1y3',
                     'delta_imp_aport_var17_1y3',
                     'delta_imp_aport_var33_1y3',
                     'delta_imp_compra_var44_1y3',
                     'delta_imp_reemb_var13_1y3',
                     'delta_imp_reemb_var17_1y3',
                     'delta_imp_reemb_var33_1y3',
                     'delta_imp_trasp_var17_in_1y3',
                     'delta_imp_trasp_var17_out_1y3',
                     'delta_imp_trasp_var33_in_1y3',
                     'delta_imp_trasp_var33_out_1y3',
                     'delta_imp_venta_var44_1y3',
                     ]
    for feature_name in feature_names:
        for df in [train, test]:
            df[feature_name] = np.where((df[feature_name] != 0)&(df[feature_name] != -1)&(df[feature_name] != 9999999999),
                               None, df[feature_name])
            df[feature_name].replace(to_replace=9999999999, value=1, inplace=True)

    features_to_drop = ['delta_num_aport_var13_1y3', # because highly correlation with other columns
                        'delta_num_aport_var17_1y3', # because highly correlation with other columns
                        'delta_num_aport_var33_1y3', # because highly correlation with other columns
                        'delta_num_compra_var44_1y3', # because highly correlation with other columns
                        'delta_num_venta_var44_1y3', # because highly correlation with other columns
                        'imp_amort_var18_ult1', # because almost no variance
                        'saldo_var18', # because almost no variance
                        'imp_amort_var34_ult1', # because almost no variance
                        'saldo_var34', # because almost no variance
                        ]
    for feature_name in features_to_drop:
        for df in [train, test]:
            df.drop(feature_name, axis=1, inplace=True)




    return train, test






def remove_linearly_dependent_features(train, test):
    remove = []
    cols = train.columns
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(train)
    train_scaled = pd.DataFrame(data=X_scaled,
                                index=range(X_scaled.shape[0]),
                                columns=train.columns)
    train_scaled['TARGET'] = train['TARGET']
    for i in range(len(cols)-1):
        v = train_scaled[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v, train_scaled[cols[j]].values):
                remove.append(cols[j])
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    return train, test

def PCA_analysis(train, test):
    variance_ratio_threshhold = 0.99
    pca = PCA()
    X = train.drop("TARGET", axis=1)
    # feature normalization
    pca.fit(X)
    components = pca.components_
    component_ratio = pca.explained_variance_ratio_
    accumulated_ratio = [sum(component_ratio[:i+1]) for i in range(len(component_ratio))]
    number_of_features_to_keep = 0
    for i in range(len(components)):
        if accumulated_ratio[i] <= variance_ratio_threshhold:
            number_of_features_to_keep += 1
    print number_of_features_to_keep

    pca = PCA(n_components=number_of_features_to_keep)
    X_new = pca.fit_transform(X)
    new_train = pd.DataFrame(data=X_new,
                             index=range(X_new.shape[0]),
                             columns=X.columns)
    new_train['TARGET'] = train['TARGET']
    return new_train


def remove_0_variant_features(train, test):
    # remove ID and features with no variance
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)
    train.drop('ID', axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    return train, test


if __name__ == "__main__":
    main()
