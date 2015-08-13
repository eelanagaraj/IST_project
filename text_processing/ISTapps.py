#! /usr/bin/env python
""" process sequence text files into a usable format """

import cPickle as pkl
import numpy as np
import os
import string
from keras.preprocessing import sequence
from sklearn import cross_validation

""" small helper for sequence extraction from string """
def has_num(string): 
	return any(c.isdigit() for c in string)


""" Takes in a text file with sequences as string and
	converts into a list of sequences (int lists)
	String format: "[[seq1], seq2,..., [seqn]]"	"""
def extract_sequences(file_path) :
	with open(file_path, 'r') as f :
		raw_text = f.read()
	# replace punctuation and convert string into int lists
	raw_text = raw_text.split(']')
	replace_punctuation = string.maketrans(string.punctuation,
		' '*len(string.punctuation))

	seq_list = [[int(x) for x in str_seq.translate(replace_punctuation).split()]
		for str_seq in raw_text if has_num(str_seq)]
	if seq_list :
		return seq_list
	else :
		return None

""" For late/early fusion stuff, each application is a seq list.
	Return a list of sequence arrays, and corresponding labels for
	each application (not for individual sentences)"""
def process_apps (pos_file_dir, neg_file_dir, maxlen) :
	pos_apps, neg_apps = [], []
	# list comp is less efficient since need to check that extract returns val
	for filename in os.listdir(pos_file_dir) :
		extracted = extract_sequences(os.path.join(pos_file_dir, filename))
		if extracted :
			# IF WANT SEQUENCES INSTEAD: extracted = sequence.pad_sequences(extracted, maxlen)
			pos_apps.append(extracted)
	for filename in os.listdir(neg_file_dir) :
		extracted = extract_sequences(os.path.join(neg_file_dir, filename))
		if extracted :
			# IF WANT SEQUENCES INSTEAD: extracted = sequence.pad_sequences(extracted, maxlen)
			neg_apps.append(extracted)
	# prepare labels per application, rather than per sequence
	y = [1]*len(pos_apps) + [0]*len(neg_apps)
	X = pos_apps + neg_apps
	return (X,y)

""" extract sequences from all text files in a directory
	return one large sequence list """
def load_files(file_dir) :
	# getting rid of punctuation in string
	replace_punctuation = string.maketrans(string.punctuation,
		' '*len(string.punctuation))
	# after timing, looping is here faster than list comprehension
	all_sequences = []
	for filename in os.listdir(file_dir) :
		filepath = os.path.join(file_dir, filename)
		extracted = extract_sequences(filepath)
		if extracted :
			all_sequences += extracted
	return all_sequences


""" create pickle file of raw, unshuffled data + labels """
def pickle_data(yes_dir, no_dir): 
	# parse all sequences
	yes_sequences = load_files(yes_dir)
	no_sequences = load_files(no_dir)
	# create labels
	yes_labels = [1]*len(yes_sequences)
	no_labels = [0]*len(no_sequences)
	X = yes_sequences + no_sequences
	y = yes_labels + no_labels
	# pickle unshuffled data and corresponding labels
	data = dict(X=X, y=y)
	pkl.dump(data, 
		open('/fs3/home/enagaraj/project/text_processing/data.pkl', 'wb'), 
		protocol=pkl.HIGHEST_PROTOCOL)

""" helper function that converts data divided by app into
	one array of sequences and corresponding label array
	** can use in main function before testing apps as well **"""
def extract_from_apps (X_prime, y_prime, maxlen, seed=107, shuffle=True) :
	# convert train set into a huge block of sequences and shuffle again
	X, y = [], []
	for i,app in enumerate(X_prime) :
		# update labels as well
		y.extend([y_prime[i]]*len(app))
		X.extend(app)
	assert (len(X) == len(y))
	# shuffle if set
	if shuffle :
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
	return (sequence.pad_sequences(X,maxlen), np.asarray(y))


"""loader function which downloads pickled data and labels, shuffles by seed,
	and pads to maximum length maxlen. Returns unsplit data,labels: (X,y)"""
def load_ISTapps (maxlen, seed=107) :
	# (process if necessary and) open stored unshuffled, unsplit data
	pkl_file = '/fs3/home/enagaraj/project/text_processing/data.pkl'
	if not(os.path.isfile(pkl_file)) : 
		pickle_data('/fs3/home/enagaraj/project/text_processing/SoP-train-yes',
			'/fs3/home/enagaraj/project/text_processing/SoP-train-no')
	raw_data = pkl.load(open(pkl_file, 'rb')) 
	X = raw_data['X']
	y = raw_data['y']

	# pad and shuffle sequences
	X = sequence.pad_sequences(X, maxlen)
	y = np.asarray(y)
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)

	return (X,y)


""" load data where each app is a data point.
	Return shuffled set of applications and corresponding labels per app. """
def load_apps_shuffled (maxlen, seed=107) :
	# load data and store for faster future access
	pkl_file = '/fs3/home/enagaraj/project/text_processing/sep_files.pkl'
	if not (os.path.isfile(pkl_file)) :
		(X,y) = process_apps('/fs3/home/enagaraj/project/text_processing/SoP-train-yes', 
			'/fs3/home/enagaraj/project/text_processing/SoP-train-no', maxlen)
		data = dict(X=X, y=y)
		pkl.dump(data, open(pkl_file, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
	else :
		data = pkl.load(open(pkl_file, 'rb'))
		X = data['X']
		y = data['y']
	# shuffle data, treating each application as a unit
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)
	return (X,y)



"""	# split into train and test sets
	split_point = int(len(X)*(1-test_split))
	X_train = X[:split_point]
	y_train = y[:split_point]
	X_test = X[split_point:]
	y_test = y[split_point:]
	# convert train set into a huge block of sequences and shuffle again
	(X_train, y_train) = extract_from_apps(X_train, y_train, seed=seed, shuffle=True)
	return (X_train, y_train), (X_test, y_test) """

(X1,y1) = load_apps_shuffled(30, 107)
(X,y) = load_ISTapps(30,107)
(X2, y2) = extract_from_apps(X1[:1], y1[:1], 30, seed=107, shuffle=True)
(X3, y3) = extract_from_apps(X1[:1], y1[:1], 30, seed=107, shuffle=False)