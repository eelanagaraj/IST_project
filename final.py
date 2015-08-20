#! /usr/bin/env python

""" Construct LSTM network. Train on the entire
	dataset and test on the validation set. """
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import cPickle as pkl

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from text_processing.ISTapps import load_ISTapps, get_file_paths
from sklearn import cross_validation
from classifiers_ISTapps import early_fusion, late_fusion

optimizer = 'adam'
LSTM_in = 256
LSTM_out = 128
activation = 'sigmoid'
loss_func = 'binary_crossentropy'
nb_epoch = 3
max_features=100000
maxlen = 50 # cut texts after this number of words
batch_size = 16
settings = (optimizer, loss_func, activation, LSTM_in, LSTM_out, nb_epoch)

print("Loading data...")
train_save_block, train_save_sep, train_yes_dir, train_no_dir = get_file_paths(train=True)
test_save_block, test_save_sep, test_yes_dir, test_no_dir = get_file_paths(train=False)
X_train, y_train = load_ISTapps(maxlen, separate=False, save_file=train_save_block,
	yes_directory=train_yes_dir, no_directory=train_no_dir, seed=111)
X_test, y_test = load_ISTapps(maxlen, separate=False, save_file=test_save_block,
	yes_directory=test_yes_dir, no_directory=test_no_dir, seed=111)

print("Settings: ", settings)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, LSTM_in))
model.add(LSTM(LSTM_in, LSTM_out))
model.add(Dropout(0.5))
model.add(Dense(LSTM_out, 1))
model.add(Activation(activation))
model.compile(loss=loss_func, optimizer=optimizer, class_mode="binary")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    validation_split=0.1, show_accuracy=True, verbose=2)
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)

classes = model.predict_classes(X_test, batch_size=batch_size)
acc = np_utils.accuracy(classes, y_test)

print('Test accuracy:', acc)
weights = model.get_weights()

with open('/home/enagaraj/final_results.txt', 'a') as f :
	print ('\navg: ', acc, 'settings: ', settings, file=f)


print('Building partial model...')
# early fusion for testing, average over each application
early_fusion_model = Sequential()
early_fusion_model.add(Embedding(max_features, LSTM_in, 
	weights=model.layers[0].get_weights()))
early_fusion_model.add(LSTM(LSTM_in, LSTM_out, 
	weights=model.layers[1].get_weights()))
early_fusion_model.compile(loss=loss_func, 
	optimizer=optimizer, class_mode="binary")

# load separate files
X_train_prime, y_train_prime = load_ISTapps(maxlen, separate=True, save_file=train_save_sep,
	yes_directory=train_yes_dir, no_directory=train_no_dir, seed=111)
X_test_prime, y_test_prime = load_ISTapps(maxlen, separate=True, save_file=test_save_sep, 
	yes_directory=test_yes_dir, no_directory=test_no_dir, seed=111)

X_train_block = early_fusion_model.predict(X_train)
X_train_rep = [early_fusion_model.predict(sequence.pad_sequences(app,maxlen)) 
	for app in X_train_prime]
X_test_rep = [early_fusion_model.predict(sequence.pad_sequences(app,maxlen))
	for app in X_test_prime]
assert(len(X_train_rep) == len(y_train_prime))
assert(len(X_test_rep) == len(y_test_prime))

early_accs=[]
late_accs =[]
for k_neighbors in [5,10,25,50,100] :
	acc_early = early_fusion(X_train_rep, y_train_prime, X_test_rep, y_test_prime, k_neighbors)
	early_accs.append(acc_early)
	acc_late = late_fusion(X_train_block, y_train, X_test_rep, y_test_prime, k_neighbors)
	late_accs.append(acc_late)
	print('k : ', k_neighbors, ', Early accuracy: ', acc_early, 'Late accuracy: ', acc_late)

