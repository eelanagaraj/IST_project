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
	# nltk's sophisticated method for detecting sentence boundaries
	sentences = detector.tokenize(raw_text.strip())
	pattern = r'''(?x)
		(([a-zA-Z]|ph)\.)+(\b[a-zA-Z]\b)?	# accronyms/abbreviations
		| \d+(\.\d+)  						# decimal numbers
		| \w+([-']\w+)*        				# words/numbers incl. hyphenated
		| \.\.\.            				# ellipsis
		| [][.,;"'?():-_`] '''				# punctuation
	# convert each sentence to a list of lowercase tokens
	sentence_list = [nltk.regexp_tokenize(sentence.lower(),pattern) 
		for sentence in sentences]
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
	return words


""" takes a text file and encodes it as a reproducibly random
	numerical sequence. The sequence cannot be converted back to
	a word sequence without the seed value. """
def parse_file (file_path, word_dict, detector, seed) :
	# read in file and get rid of markup
	with codecs.open(file_path, 'r', 'utf-8') as f :
		cleaned_sentences = clean_text(f.read(), detector)
	# randomize dictionary according to seed
	np.random.seed(seed)
	rand_dict = dict(
		zip(word_dict.keys(), np.random.permutation(word_dict.values())))
	# convert each sentence to a sequence
	sequence_list = [[rand_dict[word] for word in sentence if word in rand_dict] 
		for sentence in cleaned_sentences]
	return sequence_list


""" actual script to be run """
# load these parameters once then pass into parse function
google_100k = load_dictionary('/fs3/group/chlgrp/datasets/Google-1grams/Google-1grams-top100k.txt', 100000)
seed = np.random.randint(0, 4294967295)
sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')

# call parse_file(file_path, google_100k, sentence_detector, seed) for
# each file in t
app = parse_file('/fs3/home/enagaraj/project/test_files/768.statement_of_purpose.Eela_Nagaraj.txt', google_100k, sentence_detector, seed)



"""

rand = randomize_dict(words, test_seed)
num_conversion = {}
for word in rand :
	num_conversion[rand[word]] = words[word]
app = parse_file('/fs3/home/enagaraj/project/test_files/768.statement_of_purpose.Eela_Nagaraj.txt', words, sentence_detector, test_seed)
short_app = parse_file('/fs3/home/enagaraj/project/test_files/short_statement.txt', words, sentence_detector, test_seed)


"""
