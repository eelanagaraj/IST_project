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
	preds = [int(pos > pos_percent) for pos,neg in avg_sentence_probs]
	acc = np_utils.accuracy(preds, y_test)
	return acc, preds

