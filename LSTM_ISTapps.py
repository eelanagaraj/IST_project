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
from text_processing.ISTapps import load_ISTapps
#from ISTapps import load_ISTapps
from sklearn import cross_validation

# different structures to test out
"""
# trial 1: kept memory faulting at a certain point
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

#trial 3: try different optimizers with other settings constant
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
optimizers = [sgd, 'adagrad', 'adadelta', 'rmsprop', 'adam']
LSTM_ins = [256]
LSTM_outs = [128]
activations = ['sigmoid']
loss_functions = ['binary_crossentropy']
"""

# trial 4: try basically all combos except adadelta
sgd1dec = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sgd1 = SGD(lr=0.1, momentum=0., decay=0., nesterov=False)
sgd01 = SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
sgd001 = SGD(lr=0.001, momentum=0., decay=0., nesterov=False)

optimizers = ['sgd', sgd1, sgd01, sgd001, sgd1dec, 'adam', 'rmsprop', 'adadelta']
LSTM_in_out = [(128, 128), (128, 256), (256,128)]
activations = ['sigmoid', 'tanh', 'relu', 'softmax']
loss_functions = ['mean_squared_error', 'binary_crossentropy']


max_features=100000
maxlen = 30 # cut texts after this number of words
batch_size = 16
k = 5 # cross-validation 
#dataset_size = 15000

#results = {}
max_avg = 0
opt_settings = []
for optimizer in optimizers :
	for loss_func in loss_functions :
		for activation in activations :
			for (LSTM_in, LSTM_out) in LSTM_in_out :
				settings = (optimizer, loss_func, activation, LSTM_in, LSTM_out)
				print("Loading data...")
				(X,y) = load_ISTapps(maxlen, seed=111)

				# is there data signal ??! --> shrink dataset
				#X = X[:dataset_size]
				#y = y[:dataset_size]

				print("Settings: ", settings)
				
				# cross validation
				kfold_indices = cross_validation.KFold(len(X), n_folds=k)
				cv_round = 0
				cumulative_acc = [0]*k
				for train_indices, test_indices in kfold_indices :
					X_train = X[train_indices]
					y_train = y[train_indices]
					X_test = X[test_indices]
					y_test = y[test_indices]

					print("Cross Validation split ", cv_round)
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

					print("Train...")
					model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5,
					    validation_split=0.1, show_accuracy=True, verbose=2)
					score = model.evaluate(X_test, y_test, batch_size=batch_size)
					print('Test score:', score)

					classes = [int(val > 0.55) for val in model.predict(X_test, batch_size=batch_size)]
					#classes = model.predict_classes(X_test, batch_size=batch_size)
					acc = np_utils.accuracy(classes, y_test)

					print('Test accuracy:', acc)
					cumulative_acc[cv_round] = acc
					cv_round += 1
					
					# try to conserve some memory cause getting weird memory errors
					del X_train
					del y_train
					del X_test
					del y_test
					del model
				cross_val_acc = sum(cumulative_acc) / k
				# keep track of current maximum average and settings
				if (max_avg < cross_val_acc) :
					max_avg = cross_val_acc
					opt_settings = (settings)
				with open('/home/enagaraj/cumulative_results.txt', 'a') as f :
					print ('\nsettings: ', settings, 'accuracies: ', cumulative_acc, 'avg acc: ', cross_val_acc, file=f)
				#results[settings] = (cumulative_acc, cross_val_acc)
				print ('Average accuracy: ', cross_val_acc)
				
				# again try to satisfy the memory gods
				del X
				del y

# calculate best value
#vals = results.values()
#max_avg = 0

#for lst,avg in vals :
#	if avg > maxv :
#		max_avg = avg

with open('/home/enagaraj/cumulative_results_len30.txt', 'a') as f :
	print ('\nmax average: ', max_avg, 'optimal settings: ', opt_settings, file=f)

#print (results)