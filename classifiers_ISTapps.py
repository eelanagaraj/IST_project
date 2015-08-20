#! /usr/bin/env python

""" Early fusion with ISTapp data"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import cPickle as pkl
import os

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from text_processing import ISTapps 
from sklearn import cross_validation, neighbors

max_features=100000
maxlen = 50 # cut sentences after this number of words
batch_size = 16
k_crossval = 5 # cross-validation 
LSTM_in = 256
LSTM_out = 128
optimizer = 'adam'
loss_func = 'binary_crossentropy'
activation = 'sigmoid'
nb_epoch = 3
settings = (optimizer, loss_func, activation, nb_epoch, LSTM_in, LSTM_out)

""" construct LSTM network and train on data """
def tune_model(X_train, y_train, X_test, y_test, settings) :
	(optimizer, loss_func, activation, nb_epoch, LSTM_in, LSTM_out) = settings
	print("Loading data...")
	print(len(X_train), 'train sequences')
	print('X_train shape:', X_train.shape)

	# train LSTM so that we can extract representation
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
	classes = model.predict_classes(X_test, batch_size=batch_size)
	acc = np_utils.accuracy(classes, y_test)
	print('LSTM accuracy:', acc)

	print('Building partial model...')
	# early fusion for testing, average over each application
	early_fusion_model = Sequential()
	early_fusion_model.add(Embedding(max_features, LSTM_in, 
		weights=model.layers[0].get_weights()))
	early_fusion_model.add(LSTM(LSTM_in, LSTM_out, 
		weights=model.layers[1].get_weights()))
	early_fusion_model.compile(loss=loss_func, 
		optimizer=optimizer, class_mode="binary")
	return early_fusion_model


""" extract the partial activations of the training and test data
	from the trained neural network. Return all the activations
	of each sentence, grouped by app and for training in one block """
def extract_representation(maxlen, settings, model=None, seed=107, test_split=0.2) :
	# load data
	save_block, save_sep, yes_dir, no_dir = get_file_paths(train=True)
	(X,y) = ISTapps.load_ISTapps(maxlen, separate=True, save_file=save_block, 
		yes_directory=yes_dir, no_directory=no_dir, seed=seed)
	X = np.asarray(X)
	y = np.asarray(y)
	split_point = int(len(X)*(1-test_split))
	X_train_prime = X[:split_point]
	y_train_prime = y[:split_point]
	X_test_prime = X[split_point:]
	y_test_prime = y[split_point:]
	# convert train set into a huge block of sequences and shuffle again
	(X_train, y_train) = ISTapps.extract_from_apps(X_train_prime, 
		y_train_prime, maxlen, seed, shuffle=True)
	# convert model for LSTM success comparison
	(X_test, y_test) = ISTapps.extract_from_apps(X_test_prime, 
		y_test_prime, maxlen, seed, shuffle=True)
	if not model :
		model = tune_model(X_train, y_train, X_test, y_test, settings)
	# return training data as a shuffled sentence block and separated by app
	X_train_block = model.predict(X_train)
	X_train_rep = [model.predict(sequence.pad_sequences(app,maxlen)) 
		for app in X_train_prime]
	X_test_rep = [model.predict(sequence.pad_sequences(app,maxlen)) 
		for app in X_test_prime]
	assert(len(X_train_rep) == len(y_train_prime))
	assert(len(X_test_rep) == len(y_test_prime))

	return (X_train_block, y_train), (X_train_rep, y_train_prime), (X_test_rep, y_test_prime), model


def avg_across_sentences (X) :
	return np.asarray([np.mean(app, axis=0) for app in X])


def early_fusion (X_train_rep, y_train, X_test_rep, y_test, k_neighbors) :
	# train and test on average sentence representations of applications
	X_train = avg_across_sentences(X_train_rep)
	X_test = avg_across_sentences(X_test_rep)

	# knn
	y_train = np.asarray(y_train)
	y_test = np.asarray(y_test)
	clf = neighbors.KNeighborsClassifier(k_neighbors, weights='distance')
	clf.fit(X_train, y_train)
	acc = clf.score(X_test, y_test)
	return acc


def late_fusion (X_train, y_train, X_test, y_test, k_neighbors, seed=107) :
	# kNN
	clf = neighbors.KNeighborsClassifier(k_neighbors, weights='distance')
	clf.fit(X_train, y_train)
	threshhold = np.mean(y_train)
	# label applications based on average of sentence predictions
	app_predictions  = [int(np.mean(clf.predict(app)) > threshhold) for app in X_test]
	acc = np_utils.accuracy(app_predictions, y_test)
	return acc
