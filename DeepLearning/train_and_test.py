from __future__ import print_function
import numpy as numpy
import os
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

data_root = '.'
data_filename = os.path.join(data_root, 'notMNIST.pickle')

def read_data(file_name):
    try:
        with open(data_filename, 'rb') as f:
            merged_data = pickle.load(f)
            
    except Exception as e:
        print('Unable to read data from', data_filename, ':', e)
        raise
    return merged_data

merged_data = read_data(data_filename)

train_dataset = merged_data['train_dataset']
train_labels = merged_data['train_labels']
valid_dataset = merged_data['valid_dataset']
valid_labels = merged_data['valid_labels']
test_dataset = merged_data['test_dataset']
test_labels = merged_data['test_labels']

def convert_to_2D_dataset(dataset):
    return dataset.reshape(len(dataset), -1)


train_dataset_2D = convert_to_2D_dataset(train_dataset)
valid_dataset_2D = convert_to_2D_dataset(valid_dataset)
test_dataset_2D = convert_to_2D_dataset(test_dataset)

print('training:', train_dataset_2D.shape, train_labels.shape)
print('validation:', valid_dataset_2D.shape, valid_labels.shape)
print('test:', test_dataset_2D.shape, test_labels.shape)

train_dataset_samples = train_dataset_2D[:100]
train_labels_samples = train_labels[:100]

logisticRegression = LogisticRegression()
logisticRegression.fit(train_dataset_samples, train_labels_samples)
score = logisticRegression.score(test_dataset_2D, test_labels)
print('Score of training with validation dataset:', score)