import numpy as np
from sklearn import linear_model as lin_cf
import pickle
from sklearn.metrics import accuracy_score


def train_data(training_data, train_labelss, valid_data, valid_labelss):
    nsamples, nx, ny = training_data.shape
    d2_train_dataset = np.reshape(training_data, (nsamples, nx * ny))
    nsamples, nx, ny = valid_data.shape
    d2_valid_data = np.reshape(valid_data, (nsamples, nx * ny))
    print(d2_train_dataset.shape)
    print(train_labelss.shape)
    print(d2_valid_data.shape)
    print(valid_labelss.shape)
    clf = lin_cf.LogisticRegression()
    print('training....')
    clf.fit(d2_train_dataset, train_labelss)
    print('predicting....')
    pred = clf.predict(d2_valid_data)
    print('scoring....')
    score = accuracy_score(valid_labelss, pred)

    print(score)


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

train_dataset,train_labels,valid_dataset,valid_labels = get_data_from_pickle()
train_data(train_dataset,train_labels,valid_dataset,valid_labels)