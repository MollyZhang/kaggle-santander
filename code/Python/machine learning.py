import pandas as pd
import numpy as np
import datetime
from sklearn import cross_validation, linear_model, metrics, tree, ensemble, neighbors
import plotting


CLASSIFIERS = {#"LR": linear_model.LogisticRegression(),
               #"Dtree": tree.DecisionTreeClassifier(max_depth=20),
               #"Forest": ensemble.RandomForestClassifier(),
               # "rbf SVM": SVM.SVC(kernel="rbf"),
               # "linear SVM": SVM.SVC(kernel="linear"),
               #"KNN": neighbors.KNeighborsClassifier(),
               "linear": linear_model.LinearRegression(normalize=True)}

def main():
    df = pd.read_csv("../../data/train.csv")
    data, label = data_label_split(df)
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        data, label, test_size=0.2, train_size=0.8, random_state=0, stratify=label)
    # generate_submission(data, label)
    threshhold_result = cut_off_threshhold(CLASSIFIERS['linear'], x_train, y_train)
    plotting.plot_cutting_off(threshhold_result)

def cut_off_threshhold(clf, x_train, y_train):
    x_tr, x_val, y_tr, y_val = cross_validation.train_test_split(
        x_train, y_train, test_size=0.3, train_size=0.7, random_state=0, stratify=y_train)
    clf.fit(x_tr, y_tr)
    prediction_val = clf.predict(x_val)
    prediction_train = clf.predict(x_tr)

    row_list = []
    for threshhold in np.linspace(0.01, 0.99, num=100, endpoint=True):
        p_val = pd.Series(prediction_val >= threshhold).astype(int)
        p_train = pd.Series(prediction_train >= threshhold).astype(int)
        auc_val = metrics.roc_auc_score(y_val, p_val)
        auc_train = metrics.roc_auc_score(y_tr, p_train)
        row_list.append({'threshhold': threshhold, 'train auc': auc_train, 'validation auc': auc_val})
    result_df = pd.DataFrame(row_list)
    return result_df

def generate_submission(data, label):
    clf = CLASSIFIERS['linear']
    df_test = pd.read_csv("../../data/test.csv")
    df_test['bias'] = pd.Series(np.ones(len(df_test.index)), index=df_test.index)
    clf.fit(data, label)
    prediction = clf.predict(df_test) >= 0.5
    df_submit = pd.DataFrame()
    df_submit['ID'] = df_test['ID']
    df_submit['TARGET'] = pd.Series(prediction, index=df_test.index)
    df_submit['TARGET'] = df_submit['TARGET'].astype(int)
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    df_submit.to_csv("../../result/submission_{0}.csv".format(time), index=False)


def tenfold_cross_validation(x_train, y_train, classifiers):
    result_df = pd.DataFrame()
    foldnum = 0
    for train, val in cross_validation.StratifiedKFold(y_train, shuffle=True, n_folds=3, random_state=0):
        foldnum += 1
        print "fold %d...." %foldnum
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