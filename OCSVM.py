import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np

from sklearn.linear_model import SGDOneClassSVM
from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

#df = pd.read_csv('Apache.log',sep=' ',index_col=0)
df = pd.read_csv('jenkins_conn4.csv',sep=' ',index_col=0)

df = df.fillna('Nah')

def learning(df,nu,gamma):
    data_dummies = pd.get_dummies(df)
    #ダミーデータ確認
    #data_dummies.to_csv('data_dummies.csv')

    #ノーマルデータ抽出
    data_normal = data_dummies[data_dummies.target_train == 1]
    #テストデータ抽出
    data_test = data_dummies[data_dummies.target_test == 1]
    #外れ値データ抽出
    data_outliers = data_dummies[data_dummies.target_outlier == 1]


    #代替
    #'target_outlier', 'target_test', 'target_train'を削除してnumpy配列に
    X_train = data_normal.iloc[:,:-3].values
    X_test  = data_test.iloc[:,:-3].values
    X_outliers = data_outliers.iloc[:,:-3].values
    X_all = data_dummies.iloc[:, :-3].values

    #'target_outlier', 'target_test', 'target_train'だけnumpy配列に
    X_index = data_dummies.iloc[:, -3:].values

    transform = Nystroem(gamma=gamma)
    clf_sgd = SGDOneClassSVM(nu=nu, shuffle=True, fit_intercept=True, tol=1e-4)
    #OCSVM
    pipe_sgd = make_pipeline(transform, clf_sgd)
    pipe_sgd.fit(X_train)
    #n_correct_test is True Negative
    #n_error_test is False Positive
    #n_correct_outliers is True Positive
    #n_error_outliers is False Negative

    X_pred_train = pipe_sgd.predict(X_train)
    X_pred_test = pipe_sgd.predict(X_test)
    X_pred_outliers = pipe_sgd.predict(X_outliers)
    n_correct_train = X_pred_train[X_pred_train == 1].size
    n_error_train = X_pred_train[X_pred_train == -1].size
    n_correct_test = X_pred_test[X_pred_test == 1].size
    n_error_test = X_pred_test[X_pred_test == -1].size
    n_correct_outliers = X_pred_outliers[X_pred_outliers == -1].size
    n_error_outliers = X_pred_outliers[X_pred_outliers == 1].size
    recall = n_correct_outliers / (n_correct_outliers + n_error_outliers)
    try:
        precision = n_correct_outliers / (n_correct_outliers + n_error_test)
    except ZeroDivisionError:
        precision = 1

    
    specificity = n_correct_test / (n_correct_test + n_error_test)
    accuracy = (n_correct_test + n_correct_outliers) / (n_correct_test + n_error_test + n_correct_outliers + n_error_outliers)
    f_value = (2 * n_correct_outliers) / (2 * n_correct_outliers + n_error_test + n_error_outliers)

    print('svm.OneClassSVM(nu=' + str(nu) + ', kernel="rbf", gamma=' + str(gamma) + ')')
    print('Training Correct: ' + str(n_correct_train))
    print('Training Error: ' + str(n_error_train))
    print('True Negative: ' + str(n_correct_test))
    print('False Positive: ' + str(n_error_test))
    print('True Positive: ' + str(n_correct_outliers))
    print('False Negative: ' + str(n_error_outliers))
    print('Recall: ' + str(recall))
    print('Precision: ' + str(precision))
    print('Specificity: ' + str(specificity))
    print('Accuracy: ' + str(accuracy))
    print('F_Value: ' + str(f_value))
    print('N: ' + str(n_correct_train+n_error_train+n_correct_test+n_error_test+n_correct_outliers+n_error_outliers))
    print('')
    
    X_train_result = np.concatenate((df[df['target'] == 'train'], X_pred_train[np.newaxis, :].T), axis=1)
    X_test_result = np.concatenate((df[df['target'] == 'test'], X_pred_test[np.newaxis, :].T), axis=1)
    X_outliers_result = np.concatenate((df[df['target'] == 'outlier'], X_pred_outliers[np.newaxis, :].T), axis=1)


nu_list = [0.5,0.4,0.3,0.2,0.2,0.01]
gamma_list = [0.1, 0.01, 0.001,0.0001]


for nu in nu_list:
    for gamma in gamma_list:
        learning(df, nu, gamma)