import numpy as np
from scipy.sparse import coo_matrix  # for sparse matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score  # for evaluating results

# data path and file name
path = "data/"
train_data_fn = 'train-features.txt'
test_data_fn = 'test-features.txt'
train_label_fn = 'train-labels.txt'
test_label_fn = 'test-labels.txt'

nwords = 2500


def read_data(data_fn, label_fn):
    # read label_fn
    with open(path + label_fn) as f:
        content = f.readlines()

    label = [int(x.strip()) for x in content]

    # read data_fn
    with open(path + data_fn) as f:
        content = f.readlines()

    # # remove '\n' at the end of each line
    content = [x.strip() for x in content]

    dat = np.zeros((len(content), 3), dtype=int)

    for i, line in enumerate(content):
        a = line.split(' ')
        dat[i, :] = np.array([int(a[0]), int(a[1]), int(a[2])])

    # remember to -1 at coordinate
    data = coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)), shape=(len(label), nwords))

    return data, label


train_data, train_label = read_data(train_data_fn, train_label_fn)

test_data, test_label = read_data(test_data_fn, test_label_fn)


model = MultinomialNB()
model.fit(train_data, train_label)

test_pred = model.predict(test_data)

print('MultinomialNB: Training size = %d, accuracy = %.2f%%' %(train_data.shape[0], accuracy_score(test_label, test_pred)*100))

####################

train_data_fn = 'train-features-100.txt'
train_label_fn = 'train-labels-100.txt'
test_data_fn = 'test-features.txt'
test_label_fn = 'test-labels.txt'

train_data, train_label = read_data(train_data_fn, train_label_fn)

test_data, test_label = read_data(test_data_fn, test_label_fn)


model = MultinomialNB()
model.fit(train_data, train_label)

test_pred = model.predict(test_data)

print('MultinomialNB: Training size = %d, accuracy = %.2f%%' %(train_data.shape[0], accuracy_score(test_label, test_pred)*100))


####################
train_data_fn = 'train-features-100.txt'
train_label_fn = 'train-labels-100.txt'
test_data_fn = 'test-features.txt'
test_label_fn = 'test-labels.txt'

train_data, train_label = read_data(train_data_fn, train_label_fn)

test_data, test_label = read_data(test_data_fn, test_label_fn)


model = BernoulliNB()
model.fit(train_data, train_label)

test_pred = model.predict(test_data)

print('BernoulliNB: Training size = %d, accuracy = %.2f%%' %(train_data.shape[0], accuracy_score(test_label, test_pred)*100))

