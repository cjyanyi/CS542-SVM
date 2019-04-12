# -*- coding:UTF-8 -*-

from ps4.svm import SVC
import numpy as np
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed
import warnings
from sklearn.base import BaseEstimator,clone
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class DAGSVM:
    def __init__(self, kernel, C, n_jobs=None):
        self.kernel = kernel
        self.C = C
        self.n_jobs = n_jobs
        self.svc = SVC(kernel, C)


    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.estimators_ = {}
        n_classes = self.classes_.shape[0]
        estimators_indices = list(zip(*(Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary)
            (self.svc, X, y, self.classes_[i], self.classes_[j])
            for i in range(n_classes) for j in range(i + 1, n_classes)))))

        k=0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                self.estimators_["{}-{}".format(i, j)] = estimators_indices[0][k]
                k+=1
        node = self._create_dag_node(self.estimators_,0,n_classes-1)
        return node

    def _create_dag_node(self, svm_list, left, right):
        curr_svm = svm_list["{}-{}".format(left, right)]
        if right - left == 1:
            node = DAG_Node(curr_svm, left, right, None, None)
        else:
            left_svm = self._create_dag_node(svm_list, left + 1, right)
            right_svm = self._create_dag_node(svm_list, left, right - 1)
            node = DAG_Node(curr_svm, left, right, left_svm, right_svm)
        return node



class DAG_Node:
    def __init__(self, svc, l, r, left_svm, right_svm):
        self.svc = svc
        self.left_svc = left_svm
        self.right_svc = right_svm
        self.left = l
        self.right = r

    def predict_vector(self, x):
        predict_result = self.svc.predict(x)
        if not (self.left_svc or self.right_svc):
            return self.right if predict_result > 0 else self.left
        else:
            return self.left_svc.predict_vector(x) if predict_result > 0 else self.right_svc.predict_vector(x)

    def predict(self,X):
        return np.array([self.predict_vector(x) for x in X]).reshape(-1,1)


class _ConstantPredictor(BaseEstimator):

    def fit(self, X, y):
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat(self.y_, X.shape[0])

    def decision_function(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat(self.y_, X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat([np.hstack([1 - self.y_, self.y_])],
                         X.shape[0], axis=0)

def _fit_binary(estimator, X, y, classes=None):
    """Fit a single binary estimator."""
    # print('X shape: ',X.shape)
    # print('y shape: ', y.shape)
    # print(y)
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        y[y==0]=-1
        y.reshape(-1,1)
        estimator.fit(X, y.reshape(-1,1))
    return estimator

def _fit_ovo_binary(estimator, X, y, i, j):
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, np.int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    indcond = np.arange(X.shape[0])[cond.reshape(-1,)]
    return _fit_binary(estimator,
                       _safe_split(estimator, X, None, indices=indcond)[0],
                       y_binary, classes=[i, j]), indcond


if __name__ == '__main__':
    import scipy.io as sio
    from ps4.kernel import Kernel

    PATH = '/Users/cai/Desktop/cs 542/hw/hw4/'
    data = sio.loadmat(PATH + "MNIST_data.mat")
    train_samples = data['train_samples']
    train_samples_labels = data['train_samples_labels']
    test_samples = data['test_samples']
    test_samples_labels = data['test_samples_labels']
    svc = DAGSVM(Kernel().polynomial(5),10,n_jobs=4).fit(train_samples,train_samples_labels)
    preds = svc.predict(test_samples)

    print('DAG SVM:\n', confusion_matrix(test_samples_labels.reshape(-1, 1), preds))
    print(accuracy_score(test_samples_labels.reshape(-1, 1), preds))



