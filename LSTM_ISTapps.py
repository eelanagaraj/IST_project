#! /usr/bin/env python

""" time to run LSTM on this bad boy! """
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import cPickle as pkl

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
#from text_processing.ISTapps import load_ISTapps
from ISTapps import load_ISTapps
from sklearn import cross_validation

# different structures to test out
"""
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
optimizers = ['adam', sgd, 'adagrad', 'adadelta', 'rmsprop']
LSTM_ins = [128, 256, 512]
LSTM_outs = [128, 256]
activations = ['sigmoid', 'relu', 'softmax', 'tanh']
loss_functions = ['binary_crossentropy', 'mean_squared_error']

# trial 2: cross validation settings
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
optimizers = ['adam']
LSTM_ins = [256, 512]
LSTM_outs = [128, 256]
activations = ['sigmoid', 'relu']
loss_functions = ['binary_crossentropy']
"""

#trial 3: try different optimizers with other settings constant
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
optimizers = [sgd, 'adagrad', 'adadelta', 'rmsprop', 'adam']
LSTM_ins = [256]
LSTM_outs = [128]
activations = ['sigmoid']
loss_functions = ['binary_crossentropy']


max_features=100000
maxlen = 50 # cut texts after this number of words
batch_size = 16
k = 5 # cross-validation 

results = {}

for optimizer in optimizers :
	for loss_func in loss_functions :
		for activation in activations :
			for LSTM_in in LSTM_ins :
				for LSTM_out in LSTM_outs :
					settings = (optimizer, loss_func, activation, LSTM_in, LSTM_out)
					print("Loading data...")
					# cross validation
					(X,y) = load_ISTapps(maxlen, seed=111)
					kfold_indices = cross_validation.KFold(len(X), n_folds=k)
					cv_round = 0
					cumulative_acc = [0]*k
					for train_indices, test_indices in kfold_indices :
						X_train = X[train_indices]
						y_train = y[train_indices]
						X_test = X[test_indices]
						y_test = y[test_indices]
	
						print("Settings: ", settings)
						print(len(X_train), 'train sequences')
						print(len(X_test), 'test sequences')

						print('X_train shape:', X_train.shape)
						print('X_test shape:', X_test.shape)

						print('Build model...')
						model = Sequential()
						model.add(Embedding(max_features, LSTM_in))
						model.add(LSTM(LSTM_in, LSTM_out)) # try using a GRU instead, for fun
						model.add(Dropout(0.5))
						model.add(Dense(LSTM_out, 1))
						model.add(Activation(activation))

						# try using different optimizers and different optimizer configs
						model.compile(loss=loss_func, optimizer=optimizer, class_mode="binary")

						print ("Cross Validation split ", cv_round)
						print("Train...")
						model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5,
						    validation_split=0.1, show_accuracy=True, verbose=2)
						score = model.evaluate(X_test, y_test, batch_size=batch_size)
						print('Test score:', score)

						classes = model.predict_classes(X_test, batch_size=batch_size)
						acc = np_utils.accuracy(classes, y_test)

						print('Test accuracy:', acc)
						cumulative_acc[cv_round] = acc
						cv_round += 1
					cross_val_acc = sum(cumulative_acc) / k
					with open('/home/enagaraj/results.txt', 'a') as f :
						print ('\nsettings: ', settings, 'accuracies: ', cumulative_acc, 'avg acc: ', cross_val_acc, file=f)
					results[settings] = (cumulative_acc, cross_val_acc)
					print ('Average accuracy: ', cross_val_acc)

print (results)
#with open('/home/enagaraj/results.txt', 'a') as f :
#	print (results, file=f)
