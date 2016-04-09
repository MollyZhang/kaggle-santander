import pandas as pd
import numpy as np
from sklearn import cross_validation, linear_model, metrics, tree, ensemble


CLASSIFIERS = {#"LR": linear_model.LogisticRegression(),
               "Dtree": tree.DecisionTreeClassifier(max_depth=20),
               "Forest": ensemble.RandomForestClassifier()}


def main():

    print "data loading....."
    df = pd.read_csv("../../data/train.csv")
    data, label = data_label_split(df)
    # data = data[['var38', 'var15']]
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        data, label, test_size=0.2, train_size=0.8, random_state=0, stratify=label)

    print "training and prediction....."

    result_df = tenfold_cross_validation(x_train, y_train, CLASSIFIERS)
    print result_df.mean()


def tenfold_cross_validation(x_train, y_train, classifiers):
    result_df = pd.DataFrame()
    foldnum = 0
    for train, val in cross_validation.StratifiedKFold(y_train, shuffle=True, n_folds=3, random_state=0):
        foldnum += 1
        [tr_data, val_data, tr_targets, val_targets] = folds_to_split(x_train, y_train, train, val)
        tr_targets = tr_targets.as_matrix().ravel()
        val_targets = val_targets.as_matrix().ravel()

        for clf_name, clf in classifiers.iteritems():
            clf.fit(tr_data, tr_targets)
            auc_train = metrics.roc_auc_score(tr_targets, clf.predict(tr_data))
            auc_val = metrics.roc_auc_score(val_targets, clf.predict(val_data))
            result_df.loc[foldnum, "clf={0} train".format(clf_name)] = auc_train
            result_df.loc[foldnum, "clf={0} val".format(clf_name)] = auc_val
    return result_df


def data_label_split(df):
    label = df['TARGET']
    data = df.drop("TARGET", axis=1)
    data['bias'] = pd.Series(np.ones(len(data.index)), index=data.index)
    return data, label


def folds_to_split(data,targets,train,test):
    data_tr = pd.DataFrame(data).iloc[train]
    data_te = pd.DataFrame(data).iloc[test]
    labels_tr = pd.DataFrame(targets).iloc[train]
    labels_te = pd.DataFrame(targets).iloc[test]
    return [data_tr, data_te, labels_tr, labels_te]


if __name__ == '__main__':
    main()