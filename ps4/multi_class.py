# -*- coding:UTF-8 -*-

from ps4.svm import SVC
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import warnings

from sklearn.base import BaseEstimator,clone
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import check_is_fitted,_num_samples
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class MulticlassSVCOvOne:
    def __init__(self, kernel, C, n_jobs=None):
        self.kernel = kernel
        self.C = C
        self.n_jobs = n_jobs
        self.svc = SVC(kernel,C)

    def fit(self, X,y):
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        estimators_indices = list(zip(*(Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary)
            (self.svc, X, y, self.classes_[i], self.classes_[j])
            for i in range(n_classes) for j in range(i + 1, n_classes)))))

        self.estimators_ = estimators_indices[0]
        return self

    def predict(self, X):
        Y = self.decision_function(X)
        return self.classes_[Y.argmax(axis=1)]

    def decision_function(self, X):
        """ Decision function for the OneVsOneClassifier.
        :param X : array-like, shape = [n_samples, n_features]
        :return Y : array-like, shape = [n_samples, n_classes]
        """
        check_is_fitted(self, 'estimators_')

        Xs = [X] * len(self.estimators_)

        predictions = np.vstack([est.predict(Xi).reshape(-1,)
                                 for est, Xi in zip(self.estimators_, Xs)]).T
        confidences = np.vstack([_predict_binary(est, Xi)
                                 for est, Xi in zip(self.estimators_, Xs)]).T
        Y = _decision_function_ovr(predictions,
                                   confidences, len(self.classes_))
        return Y




class MulticlassSVCOvAll:
    def __init__(self, kernel, C, n_jobs=None):
        self.kernel = kernel
        self.C = C
        self.n_jobs = n_jobs
        self.svc = SVC(kernel,C)

    def fit(self, X, y):
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        self._build_estimators(X,columns)
        return self

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        n_samples = _num_samples(X)
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)
        for i, e in enumerate(self.estimators_):
            pred = _predict_binary(e, X)
            np.maximum(maxima, pred, out=maxima)
            argmaxima[maxima == pred] = i
        return self.classes_[np.array(argmaxima.T)]


    def _build_estimators(self,X,columns):
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
            self.svc, X, column, classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]])
            for i, column in enumerate(columns))

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

def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    try:
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        score = estimator.predict_proba(X)[:, 1]
    return score

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

def _decision_function_ovr(predictions, confidences, n_classes):
    '''
    :param predictions: array-like, shape (n_samples, n_classifiers)
    :param confidences: array-like, shape (n_samples, n_classifiers)
    :param n_classes:
    :return:
    '''
    n_samples = predictions.shape[0]
    votes = np.zeros((n_samples, n_classes))
    sum_of_confidences = np.zeros((n_samples, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sum_of_confidences[:, i] -= confidences[:, k]
            sum_of_confidences[:, j] += confidences[:, k]
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1

    max_confidences = sum_of_confidences.max()
    min_confidences = sum_of_confidences.min()

    if max_confidences == min_confidences:
        return votes

    # Scale the sum_of_confidences to (-0.5, 0.5) and add it with votes.
    # The motivation is to use confidence levels as a way to break ties in
    # the votes without switching any decision made based on a difference
    # of 1 vote.
    eps = np.finfo(sum_of_confidences.dtype).eps
    max_abs_confidence = max(abs(max_confidences), abs(min_confidences))
    scale = (0.5 - eps) / max_abs_confidence
    return votes + sum_of_confidences * scale


if __name__ == '__main__':
    #%%
    print('start')
    import scipy.io as sio
    from ps4.kernel import Kernel
    PATH = '/Users/cai/Desktop/cs 542/hw/hw4/'
    data = sio.loadmat(PATH+"MNIST_data.mat")
    train_samples = data['train_samples']
    train_samples_labels = data['train_samples_labels']
    test_samples = data['test_samples']
    test_samples_labels = data['test_samples_labels']
    test_samples_labels.shape

    #%%
    print('MulticlassSVC OvOne:')
    svc = MulticlassSVCOvOne(Kernel().polynomial(5), 10,n_jobs=4)
    svc.fit(train_samples,train_samples_labels)
    #%%
    preds = svc.predict(test_samples)
    # print(confusion_matrix(test_samples_labels.reshape(-1,1),preds))
    # print(accuracy_score(test_samples_labels.reshape(-1,1),preds))

    #%%
    print('MulticlassSVC OvAll:')
    svc = MulticlassSVCOvAll(Kernel().rbf(0.5), 50, n_jobs=4)
    svc.fit(train_samples, train_samples_labels)
    preds2 = svc.predict(test_samples)

    print('MulticlassSVC OvOne:\n',confusion_matrix(test_samples_labels.reshape(-1, 1), preds))
    print(accuracy_score(test_samples_labels.reshape(-1,1),preds))
    print('MulticlassSVC OvAll:\n',confusion_matrix(test_samples_labels.reshape(-1, 1), preds2))
    print(accuracy_score(test_samples_labels.reshape(-1, 1), preds2))

