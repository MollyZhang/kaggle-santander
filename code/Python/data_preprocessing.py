import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def main():
    train = pd.read_csv("../../data/train_4-26.csv")
    test = pd.read_csv("../../data/test_4-26.csv")
    train, test = feature_engineering(train, test)
    train.to_csv('../../data/train_4-28.csv', index=False)
    test.to_csv('../../data/test_4-28.csv', index=False)

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
                               2, df[feature_name])
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
                        'imp_aport_var33_ult1', # because almost no variance
                        'imp_op_var39_efect_ult1', # because highly correlation with other columns
                        'imp_op_var39_efect_ult3', # because highly correlation with other columns
                        'imp_op_var39_ult1', # because highly correlation with other columns
                        'imp_op_var41_comer_ult1', # highly correlated
                        'imp_op_var41_comer_ult3', # higly correlated
                        'imp_reemb_var17_hace3', # almost no variance
                        'imp_trasp_var17_in_hace3', #almost no variance
                        'imp_trasp_var17_in_ult1', #almost no variance
                        'imp_trasp_var17_out_ult1', #almost no varaince
                        'imp_trasp_var33_in_hace3', #almost no variance
                        'imp_trasp_var33_in_ult1', #almost no variance
                        'imp_var7_emit_ult1', #almost no variance
                        'imp_venta_var44_hace3', #almost no variance
                        'ind_var13_medio_0', #almost no variance
                        'ind_var18_0', #almost no variance
                        'ind_var34_0', #almost no variance
                        'ind_var40', # high correlation
                        'ind_var40_0', # high correlation
                        'ind_var6', # almost no variance
                        'ind_var6_0', # almost no variance
                        'ind_var7_emit_ult1', # almost no variance
                        'num_meses_var29_ult3', #almost no variance
                        'num_reemb_var17_hace3', # almost no variance
                        'num_trasp_var17_in_hace3', # almost no variance
                        'num_trasp_var17_in_ult1', # almost no variance, highly correlated
                        'num_trasp_var33_in_hace3', #almost no variance, highly correlated
                        'num_trasp_var33_in_ult1', #almost no variance, highly correlated
                        'num_var40_0', # highly corelated
                        'saldo_medio_var29_hace2', #almost no variance
                        'saldo_medio_var29_hace3', #almost no variance
                        ]
    for feature_name in features_to_drop:
        for df in [train, test]:
            df.drop(feature_name, axis=1, inplace=True)


    features_left_alone = ['imp_aport_var13_hace3', # amount for contribution 3 ago
                           'imp_aport_var13_ult1',
                           'imp_aport_var17_hace3',
                           'imp_aport_var17_ult1',
                           'imp_aport_var33_hace3',
                           'imp_compra_var44_hace3',
                           'imp_compra_var44_ult1',
                           'imp_ent_var16_ult1',
                           'imp_op_var39_comer_ult1',
                           'imp_op_var39_comer_ult3',
                           'imp_op_var40_comer_ult1',
                           'imp_op_var40_comer_ult3',
                           'imp_op_var40_efect_ult1', # thess two really similar
                           'imp_op_var40_efect_ult3', # these two really similar
                           'imp_op_var40_ult1',
                           'imp_op_var41_efect_ult1',
                           'imp_op_var41_efect_ult3',
                           'imp_op_var41_ult1',
                           'imp_reemb_var13_ult1', # mostly zeros
                           'imp_reemb_var17_ult1', # mostly zeros
                           'imp_sal_var16_ult1',
                           'imp_trans_var37_ult1',
                           'imp_var43_emit_ult1',
                           'imp_var7_recib_ult1',
                           'imp_venta_var44_ult1',
                           'ind_var1', # kinda nice feature
                           'ind_var10_ult1', # kinda nice
                           'ind_var10cte_ult1', # kinda nice
                           'ind_var12', # nice
                           'ind_var12_0', # nice
                           'ind_var13', #nice
                           'ind_var13_0', #nice
                           'ind_var13_corto',
                           'ind_var13_corto_0',
                           'ind_var13_largo',
                           'ind_var13_largo_0',
                           'ind_var14',
                           'ind_var14_0'
                           'ind_var17',
                           'ind_var17_0',
                           'ind_var19',
                           'ind_var1_0',
                           'ind_var20',
                           'ind_var20_0',
                           'ind_var24',
                           'ind_var24_0',
                           'ind_var25_0',
                           'ind_var25_cte',
                           'ind_var26_0',
                           'ind_var26_cte',
                           'ind_var30', # nice
                           'ind_var31',
                           'ind_var31_0',
                           'ind_var32_0',
                           'ind_var32_cte',
                           'ind_var33',
                           'ind_var33_0',
                           'ind_var37_0',
                           'ind_var37_cte',
                           'ind_var39_0', #nice
                           'ind_var41_0',
                           'ind_var43_emit_ult1', #nice
                           'ind_var43_recib_ult1', #nice
                           'ind_var44',
                           'ind_var44_0',
                           'ind_var5', # nice
                           'ind_var5_0', # nice
                           'ind_var7_recib_ult1',
                           'ind_var8',
                           'ind_var8_0',
                           'ind_var9_cte_ult1',
                           'ind_var9_ult1',
                           'num_aport_var13_hace3', #nice
                           'num_aport_var13_ult1',
                           'num_aport_var17_hace3',
                           'num_aport_var17_ult1',
                           'num_aport_var33_hace3',
                           'num_aport_var33_ult1', #mostly zeros
                           'num_compra_var44_hace3',
                           'num_compra_var44_ult1',
                           'num_ent_var16_ult1',
                           'num_med_var22_ult3',
                           'num_med_var45_ult3', #nice
                           'num_meses_var12_ult3',
                           'num_meses_var13_corto_ult3',
                           'num_meses_var13_largo_ult3',
                           'num_meses_var17_ult3',
                           'num_meses_var33_ult3',
                           'num_meses_var39_vig_ult3', #very nice
                           'num_meses_var44_ult3',
                           'num_meses_var5_ult3',
                           'num_meses_var8_ult3',
                           'num_op_var39_comer_ult1',
                           'num_op_var39_comer_ult3',
                           'num_op_var39_efect_ult1',
                           'num_op_var39_efect_ult3',
                           'num_op_var39_hace2',
                           'num_op_var39_hace3',
                           'num_op_var39_ult1',
                           'num_op_var39_ult3',
                           'num_op_var40_comer_ult1',
                           'num_op_var40_comer_ult3',
                           'num_op_var40_efect_ult1', # multiply of 3
                           'num_op_var40_efect_ult3', # multiply of 3
                           'num_op_var40_hace2', # multiply of 3
                           'num_op_var40_hace3', # multiply of 3
                           'num_op_var40_ult1', # multiply of 3
                           'num_op_var40_ult3', # multiply of 3
                           'num_op_var41_comer_ult1', #multiply of 3
                           'num_op_var41_comer_ult3', # multiply of 3
                           'num_op_var41_efect_ult1', #multiple of 3
                           'num_op_var41_efect_ult3', # multiple of 3
                           'num_op_var41_hace2', # multiple of 3
                           'num_op_var41_hace3', # multiple of 3
                           'num_op_var41_ult1', # multiple of 3
                           'num_op_var41_ult3', # multiple of 3
                           'num_reemb_var17_ult1', #mostly 0, multiple of 3
                           'num_sal_var16_ult1', # multiple of 3
                           'num_trasp_var11_ult1', # multiple of 3
                           'num_var1', # highly correlated
                           'num_var12', # multiple of 3
                           'num_var12_0', # multiple of 3
                           'num_var13', # multiple of 3
                           'num_var13_0',
                           'num_var13_corto',
                           'num_var13_corto_0',
                           'num_var13_largo',
                           'num_var13_largo_0',
                           'num_var14',
                           'num_var14_0',
                           'num_var17',
                           'num_var17_0',
                           'num_var1_0',
                           'num_var22_hace2', #nice
                           'num_var22_hace3', # nice
                           'num_var22_ult1', # nice
                           'num_var22_ult3', # nice
                           'num_var24',
                           'num_var24_0',
                           'num_var25_0',
                           'num_var26_0',
                           'num_var30',
                           'num_var30_0', # interesting, not 0 mode anymore, 3 mode
                           'num_var31', # interesting, not 0 mode anymore, 3 mode
                           'num_var31_0',
                           'num_var33_0',
                           'num_var33',
                           'num_var33_0',
                           'num_var35',
                           'num_var37_0',
                           'num_var37_med_ult2',
                           'num_var39_0',
                           'num_var4',
                           'num_var41_0',
                           'num_var42', #nice
                           'num_var42_0',
                           'num_var43_emit_ult1',
                           'num_var43_recib_ult1',
                           'num_var44_0',
                           'num_var45_hace2',
                           'num_var45_hace3',
                           'num_var45_ult1',
                           'num_var45_ult3',
                           'num_var5', # intereting, seem to be redundant of ind_var5
                           'num_var5_0',
                           'num_var7_recib_ult1',
                           'num_var8_0',
                           'num_venta_var44_hace3',
                           'saldo_medio_var12_hace2',
                           'saldo_medio_var12_hace3',
                           'saldo_medio_var12_ult1',
                           'saldo_medio_var12_ult3',
                           'saldo_medio_var13_corto_hace2',
                           'saldo_medio_var13_corto_hace3',
                           'saldo_medio_var13_corto_ult1',
                           'saldo_medio_var13_corto_ult3',
                           'saldo_medio_var13_largo_hace2',
                           'saldo_medio_var13_largo_hace3',
                           'saldo_medio_var13_largo_ult1',
                           'saldo_medio_var13_largo_ult3',
                           'saldo_medio_var13_medio_hace2',
                           'saldo_medio_var13_medio_hace2',
                           'saldo_medio_var17_hace2',
                           'saldo_medio_var17_hace3',
                           'saldo_medio_var17_ult1',
                           'saldo_medio_var17_ult3',
                           'saldo_medio_var29_ult1',
                           'saldo_medio_var29_ult3',
                           'saldo_medio_var33_hace2', # real values
                           'saldo_medio_var33_hace3', # real values
                           'saldo_medio_var33_ult1',
                           'saldo_medio_var33_ult3',
                           'saldo_medio_var44_hace2',
                           'saldo_medio_var44_hace3', # real values
                           'saldo_medio_var44_ult1',
                           'saldo_medio_var44_ult1',
                           'saldo_medio_var5_hace2', # interesting
                           'saldo_medio_var5_hace3', # interesting
                           'saldo_medio_var5_ult1',
                           'saldo_medio_var5_ult3',
                           'saldo_medio_var8_hace2',
                           'saldo_medio_var8_hace3',
                           'saldo_medio_var8_ult1',
                           'saldo_medio_var8_ult1',
                           'saldo_var1',
                           'saldo_var12',
                           'saldo_var13',
                           'saldo_var13_corto',
                           'saldo_var13_largo',
                           'saldo_var13_medio',
                           'saldo_var14',
                           'saldo_var17',
                           'saldo_var20',
                           'saldo_var25',
                           'saldo_var26',
                           'saldo_var30',
                           'saldo_var31',
                           'saldo_var32',
                           'saldo_var33',
                           'saldo_var34',
                           'saldo_var37',
                           'saldo_var40',
                           'saldo_var42',
                           'saldo_var44',
                           'saldo_var5',
                           'saldo_var6',
                           'saldo_var8',
                           'var15', # very nice
                           'var21',
                           'var3',
                           'var36',
                           'var38'
                           ]


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
