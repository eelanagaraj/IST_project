#! /usr/bin/env python
# -*- coding: utf-8 -*-


""" Converts text files into randomized sequences which
	cannot be translated without the seed value. """

import cPickle as pkl
import numpy as np
import os
import nltk
import nltk.data
import codecs
import sys


""" take in raw text, converts each sentence to list of lowercase
	words and returns list of sentence word lists."""
def clean_text (raw_text, detector) :
	raw_text = raw_text.decode(encoding='UTF-8')
	# nltk's sophisticated method for detecting sentence boundaries
	sentences = detector.tokenize(raw_text.strip())
	pattern = r'''(?x)
		(([a-zA-Z]|ph)\.)+(\b[a-zA-Z]\b)?	# accronyms/abbreviations
		| \d+(\.\d+)  						# decimal numbers
		| \w+([-']\w+)*        				# words/numbers incl. hyphenated
		| \.\.\.            				# ellipsis
		| [][.,;"'?():-_`] '''				# punctuation
	# convert each sentence to a list of lowercase tokens
	sentence_list = []
	for sentence in sentences :
		text = sentence.lower()
		word_lst = nltk.regexp_tokenize(text, pattern)
		sentence_list.append(word_lst)
	return sentence_list


""" Takes in a dictionary text file and maps each word to a number
	based on frequency. Returns this dictionary of mappings. """
def load_dictionary(file_path, max_index) :
	words = {}
	with codecs.open(file_path, 'r', 'utf-8') as f :
		count = max_index
		for line in f :
			l = line.split(' ')
			words[l[0]] = count
			count -= 1
	# save as pickle file???
	return words


""" takes a text file and encodes it as a reproducibly random
	numerical sequence. The sequence cannot be converted back to
	a word sequence without the seed value. """
def parse_file (data, word_dict, detector, seed) :
	# MAKE SURE DATA IS UTF-8 ENCODED!!!
	cleaned_sentences = clean_text(data, detector)
	# randomize dictionary according to seed
	np.random.seed(seed)
	rand_dict = dict(
		zip(word_dict.keys(), np.random.permutation(word_dict.values())))
	# convert each sentence to a sequence
	sequence_list = []
	for sentence in cleaned_sentences :
		sequence = []
		for word in sentence :
			# if word not in dictionary don't include
			if word in rand_dict :
				sequence.append(rand_dict[word])
		sequence_list.append(sequence)
	sys.stdout.write(str(sequence_list))
	#print sequence_list


""" actual script to be run """
# load these parameters once, then pass into parse function
google_100k = load_dictionary(
	'/fs3/group/chlgrp/datasets/Google-1grams/Google-1grams-top100k.txt', 100000)
seed = np.random.randint(0, 4294967295)
detector = nltk.data.load('tokenizers/punkt/english.pickle')

#data = sys.stdin.read()
# parse_file(data, google_100k, detector, seed)

