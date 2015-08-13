#! /usr/bin/env python

""" naive bayes classifier to get a baseline"""
import cPickle as pkl
import numpy as np
from text_processing import ISTapps 
from keras.utils import np_utils
from sklearn import cross_validation

""" compute ratios of words occurring in positive vs negative samples
	X is a list or matrix of sequences; e.g. each sentence has label """
def compute_ratios (X, y, max_features) :
	# array storing number of (pos,neg) occurrences
	occurrences = [(0,0)]*max_features
	for i,sentence in enumerate(X) :
		for word_index in sentence :
			# update occurrences
			pos,neg = occurrences[word_index]
			if y[i] :
				occurrences[word_index] = (pos + 1, neg)
			else :
				occurrences[word_index] = (pos, neg + 1)
	# compute ratios
	ratios = [(pos/(pos + neg* 1.0), neg/(pos + neg * 1.0)) if (pos + neg) 
				else (0., 0.) for pos,neg in occurrences]	
	return ratios


""" compute the average pos/neg probabilities of words in a sentence
	return (positive average, negative average) """
def avg_word_probabilities (sentence, ratios) :
	avg_pos = 0.
	avg_neg = 0.
	for word_index in sentence :
		avg_pos += ratios[word_index][0]
		avg_neg += ratios[word_index][1]
	avg_pos = avg_pos / len(sentence)
	avg_neg = avg_neg / len(sentence)
	return (avg_pos, avg_neg)


""" run the classifier and see how the dataset does without LSTM """
def naive_bayes (X_train, y_train, X_test, y_test) :
	# compute ratios
	ratios = compute_ratios(X_train, y_train, max_features)
	pos_percent = sum(y_train) / (len(y_train) * 1.0)
	# predict on test data based on occurrence ratios
	avg_sentence_probs = [avg_word_probabilities(sentence, ratios) for sentence in X_test]
	preds_55 = [int(pos > neg and pos > 0.55) for pos,neg in avg_sentence_probs]
	preds_percent = [int(pos > neg and pos > pos_percent) for pos,neg in avg_sentence_probs]
	acc55 = np_utils.accuracy(preds_55, y_test)
	accperc = np_utils.accuracy(preds_percent, y_test)
	return acc55, accperc, preds_55, preds_percent, avg_sentence_probs, ratios, pos_percent

test_split = 0.2
maxlen = 50
max_features = 100000
k = 5
seeds = range(1, 100)
seed_avgs_55 = [0]*len(seeds)
seed_avgs_perc = [0]*len(seeds)
for seed in seeds :
	# cross validation
	X,y = ISTapps.load_ISTapps(maxlen, seed=seed)
	X = np.asarray(X)
	y = np.asarray(y)

	kfold_indices = cross_validation.KFold(len(X), n_folds=k)
	cv_round = 0
	cumulative_acc_55 = [0]*k
	cumulative_acc_perc = [0]*k
	for train_indices, test_indices in kfold_indices :
		X_train = X[train_indices]
		y_train = y[train_indices]
		X_test = X[test_indices]
		y_test = y[test_indices]

		acc55,accperc,preds_55, preds_percent, avg_sentence_probs,ratios,pos_percent = naive_bayes(X_train, y_train, X_test, y_test)
		cumulative_acc_55[cv_round] = acc55
		cumulative_acc_perc[cv_round] = accperc

		cv_round += 1
	seed_avgs_55[seed-1] = sum(cumulative_acc_55) / k
	seed_avgs_perc[seed-1] = sum(cumulative_acc_perc) / k

# try multiplying all pos occurrence ratios to find 
# maybe eliminate ratios for words that are too popular ??? e.g. top 100 most common words or something?
