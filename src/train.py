import numpy as np
from sklearn import linear_model as lin_cf
import pickle
from sklearn.metrics import accuracy_score


def train_data(training_data, train_labelss, valid_data, valid_labelss, c, learner):
    nsamples, nx, ny = training_data.shape
    d2_train_dataset = np.reshape(training_data, (nsamples, nx * ny))
    nsamples, nx, ny = valid_data.shape
    d2_valid_data = np.reshape(valid_data, (nsamples, nx * ny))
    print(d2_train_dataset.shape)
    print(train_labelss.shape)
    print(d2_valid_data.shape)
    print(valid_labelss.shape)
    clf = lin_cf.LogisticRegression(C=c, solver=learner)
    print('training....')
    clf.fit(d2_train_dataset, train_labelss)
    print('predicting....')
    pred_tr = clf.predict(d2_train_dataset)
    pred_cv = clf.predict(d2_valid_data)
    print('scoring....')
    score_tr = accuracy_score(train_labelss, pred_tr)
    score_cv = accuracy_score(valid_labelss, pred_cv)

    return score_tr, score_cv


def get_data_from_pickle():
    try:
        with open("C:/Users/ketan_5dx9i0d/karyaghar/mlprac/data/notMNIST/AtoE.pickle", "rb") as f:
            data = pickle.load(f)
            return data.get('train_dataset'), data.get('train_labels'),\
                   data.get('valid_dataset'), data.get('valid_labels')
            return load
    except Exception as e:
        print('Error saving file', e)
        raise


def try_all():
    resultList=[]
    train_dataset, train_labels, valid_dataset, valid_labels = get_data_from_pickle()
    score_train, score_cv = train_data(train_dataset[:10000], train_labels[:10000], valid_dataset, valid_labels, 0.33, 'sag')
    resultList.append(['sag cv = 0.33', score_train, score_cv])
    score_train, score_cv = train_data(train_dataset[:10000], train_labels[:10000], valid_dataset, valid_labels, 0.33, 'newton-cg')
    resultList.append(['newtoncg cv = 0.33', score_train, score_cv])
    score_train, score_cv = train_data(train_dataset[:10000], train_labels[:10000], valid_dataset, valid_labels, 0.33, 'lbfgs')
    resultList.append(['libfgs cv = 0.33', score_train, score_cv])
    score_train, score_cv = train_data(train_dataset[:10000], train_labels[:10000], valid_dataset, valid_labels, 0.15, 'sag')
    resultList.append(['sag cv = 0.15', score_train, score_cv])
    score_train, score_cv = train_data(train_dataset[:10000], train_labels[:10000], valid_dataset, valid_labels, 0.15, 'newton-cg')
    resultList.append(['newtoncg cv = 0.15', score_train, score_cv])
    score_train, score_cv = train_data(train_dataset[:10000], train_labels[:10000], valid_dataset, valid_labels, 0.15, 'lbfgs')
    resultList.append(['libfgs cv = 0.15', score_train, score_cv])
    """score_train, score_cv = train_data(train_dataset[:10000], train_labels[:10000], valid_dataset, valid_labels, 0.5,
                                       'sag')
    resultList.append(['sag cv = 1.5', score_train, score_cv])
    score_train, score_cv = train_data(train_dataset[:10000], train_labels[:10000], valid_dataset, valid_labels, 0.5,
                                       'newton-cg')
    resultList.append(['newtoncg cv = 1.5', score_train, score_cv])
    score_train, score_cv = train_data(train_dataset[:10000], train_labels[:10000], valid_dataset, valid_labels, 0.5,
                                       'lbfgs')
    resultList.append(['libfgs cv = 1.5', score_train, score_cv])"""
    print(resultList)


try_all()
# with 100k training & 5k validation
# default: 0.84
#
#
#
# with 10k training & 5k validation
# [['sag c = 1', 0.91520000000000001, 0.84760000000000002],
#  ['newtoncg c = 1', 0.9153, 0.84719999999999995],
# ['libfgs c = 1', 0.91379999999999995, 0.84799999999999998],
#  ['sag c = 0.5', 0.91139999999999999, 0.85260000000000002],
# ['newtoncg c = 0.5', 0.91139999999999999,#  0.85260000000000002],
#  ['libfgs c = 0.5', 0.91090000000000004, 0.8528]]
#
# [['sag c = 0.33', 0.90790000000000004, 0.85560000000000003],
#  ['newtoncg c = 0.33', 0.90790000000000004, 0.85560000000000003],
#  ['libfgs c = 0.33', 0.90759999999999996, 0.85560000000000003],
#  ['sag c = 0.15', 0.90029999999999999, 0.86119999999999997],
#  ['newtoncg c = 0.15', 0.90029999999999999, 0.86119999999999997],
#  ['libfgs c = 0.15', 0.90059999999999996, 0.86119999999999997]]

# as we see, it's mostly dependent on c, lower c lower traing score and higher validation score
# 



