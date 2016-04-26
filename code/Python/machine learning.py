import pandas as pd
import numpy as np
import datetime
import pickle

from sklearn import cross_validation, linear_model, svm, metrics, tree, ensemble, neighbors
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb


import plotting


CLASSIFIERS = {#"LR": linear_model.LogisticRegression(),
               #"Dtree": tree.DecisionTreeClassifier(max_depth=20),
               #"Forest": ensemble.RandomForestClassifier(),
               #"rbf SVM": svm.SVC(kernel="rbf"),
               # "linear SVM": SVM.SVC(kernel="linear"),
               #"KNN": neighbors.KNeighborsClassifier(),
               "xgboost": xgb.XGBClassifier(n_estimators=50),
               }

def main():
    df = pd.read_csv("../../data/train_4-26.csv")
    data, label = data_label_split(df)
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        data, label, test_size=0.2, train_size=0.8, random_state=0, stratify=label)
    learning_curve(CLASSIFIERS, data, label)

    # result = tenfold_cross_validation(x_train, y_train, CLASSIFIERS)
    # print result
    #generate_submission(data, label)

def tenfold_cross_validation(x_train, y_train, classifiers):
    result_df = pd.DataFrame()
    foldnum = 0
    for train, val in cross_validation.StratifiedKFold(y_train, shuffle=True, n_folds=1, random_state=0):
        foldnum += 1
        print "fold %d...." %foldnum
        [tr_data, val_data, tr_targets, val_targets] = folds_to_split(x_train, y_train, train, val)
        tr_targets = tr_targets.as_matrix().ravel()
        val_targets = val_targets.as_matrix().ravel()
        for clf_name, clf in classifiers.iteritems():
            clf.fit(tr_data, tr_targets, eval_metric='auc')
            predictions_train = clf.predict_proba(tr_data)[:,1]
            predictions_val = clf.predict_proba(val_data)[:,1]
            auc_train = metrics.roc_auc_score(tr_targets, predictions_train)
            auc_val = metrics.roc_auc_score(val_targets, predictions_val)
            result_df.loc[foldnum, "clf={0} train".format(clf_name)] = auc_train
            result_df.loc[foldnum, "clf={0} val".format(clf_name)] = auc_val
    return result_df

def learning_curve(classifiers, data, label):
    x_tr, x_val, y_tr, y_val = cross_validation.train_test_split(
        data, label, test_size=0.2, train_size=0.8, random_state=20160426, stratify=label)
    num_samples = x_tr.shape[0]
    step = 5000
    total_samples = 1000
    result_df = pd.DataFrame()
    while total_samples <= num_samples:
        print total_samples
        clf = classifiers['xgboost']
        this_x_tr = x_tr.iloc[:total_samples,:]
        this_y_tr = y_tr.iloc[:total_samples]

        clf.fit(this_x_tr, this_y_tr, eval_metric='auc')
        predictions_train = clf.predict_proba(this_x_tr)[:,1]
        predictions_val = clf.predict_proba(x_val)[:,1]
        auc_train = metrics.roc_auc_score(this_y_tr, predictions_train)
        auc_val = metrics.roc_auc_score(y_val, predictions_val)
        result_df.loc[total_samples, "train"] = auc_train
        result_df.loc[total_samples, "val"] = auc_val
        total_samples += step
    print result_df
    f = open('pickle_dumps', "w")
    f.write(pickle.dumps(result_df))
    f.close()


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
    # plotting.plot_cutting_off(threshhold_result)
    # print max(threshhold_result['validation auc'])
    return result_df

def generate_submission(data, label):
    clf = CLASSIFIERS['xgboost']
    df_test = pd.read_csv("../../data/test_4-26.csv")
    df_test_no_id = df_test.drop('ID', axis=1)
    clf.fit(data, label, eval_metric='auc')
    prediction = clf.predict_proba(df_test_no_id)[:,1]
    for i in range(len(prediction)):
        if prediction[i] < 0:
            prediction[i] = 0
        if prediction[i] > 1:
            prediction[i] = 1
    df_submit = pd.DataFrame()
    df_submit['ID'] = df_test['ID']
    df_submit['TARGET'] = pd.Series(prediction, index=df_test.index)
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    df_submit.to_csv("../../result/submission_{0}.csv".format(time), index=False)


def data_label_split(df):
    label = df['TARGET']
    data = df.drop("TARGET", axis=1)
    return data, label


def folds_to_split(data,targets,train,test):
    data_tr = pd.DataFrame(data).iloc[train]
    data_te = pd.DataFrame(data).iloc[test]
    labels_tr = pd.DataFrame(targets).iloc[train]
    labels_te = pd.DataFrame(targets).iloc[test]
    return [data_tr, data_te, labels_tr, labels_te]


if __name__ == '__main__':
    main()