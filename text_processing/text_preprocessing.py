#! /usr/bin/env python
# -*- coding: utf-8 -*-

# look into shebang utf8 stuff ^^ --> proper string/weird character execution

""" parse text files --> need to make sure to maintain
	privacy? enough abstraction or something? 

	train dictionary and process sequences and everything

	this script should maybe just return the pickled 
	sequence matrices; that would serve to maintain anonymity?? """

import cPickle as pkl
import numpy as np
import os
import nltk
import re

from keras.preprocessing import sequence, text
from keras.utils import np_utils
from keras.datasets import imdb
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 


""" take in raw text, returns unicode text without html markup, etc"""
def clean_text (raw_text) :
    text = BeautifulSoup(raw_text, "html.parser").get_text()    
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    word_lst = letters_only.lower().split()                             
    return word_lst
   #return (" ".join(words))

# todo:
# function to run one-time to produce google dictionary pkl file
# open 100k 1gram word file, line by line go through and decrement (reverse order, make sure most common words = lower nums)
# 	 and build up word to sequence number dictionary
#	 should return the pickled dictionary
# as of now: does not handle special characters, so I guess they will potentially be ignored in actual problem?
	# shouldn't be hugely important in this case
def load_dictionary(file_path, max_index) :
	words = {}
	with open(file_path, 'r') as f :
		count = max_index
		for line in f :
			l = line.split(' ')
			words[l[0]] = count
			count -= 1
	# save as pickle file???
	return words

#test
words = load_dictionary('/fs3/group/chlgrp/datasets/Google-1grams/Google-1grams-top100k.txt', 100000)
small_words = load_dictionary('/fs3/home/enagaraj/project/test_files/test_dict.txt', 36)


# put it all together and have the function loop through a large list of strings,
	# though potentially just keep it handling single files??
	# will this be possible with tokenizer stuff? well I guess produce a script
		# have a script that initializes this tokenizer, then calls the function above 
		# for each file or whatever. Potentially take in the tokenizer argument then and
		# then just pass in this tokenizer object each time??
		# overall produce one sequence matrix?

# function to randomize indices in dictionary --> preserve anonymity
""" given a seed, shuffles the indices that correspond with the 
	words. This function is to be called within the text parsing
	function. The seed should be randomly generated once before
	processing all the code. """
def randomize_dict (word_dict, seed) :
	np.random.seed(seed)
	rand_dict = dict(
		zip(word_dict.keys(), np.random.permutation(word_dict.values())))
	return rand_dict


# TO FIX POTENTIALLY: is there an alternative to re-constructing this dict each time?
# it seems like this is the only way to ensure that someone really could not convert
# from the randomized text back to text ???
def parse_file (file_path, word_dict, seed) :
	# read in file and get rid of markup
	with open(file_path, 'r') as f :
		cleaned = clean_text(f.read())
	# randomize dictionary according to seed
	np.random.seed(seed)
	rand_dict = dict(
		zip(word_dict.keys(), np.random.permutation(word_dict.values())))
	# convert word list to sequence
	sequence = []
	for word in cleaned :
		# if word not in dictionary don't include
		if word in rand_dict :
			sequence.append(rand_dict[word])
	return sequence



# testing --> convert randomized seq to the same seq outputted by tokenizer

# fix random seed generator and test to make sure still works
test_seed = os.urandom(10)
rand = randomize_dict(words, test_seed)
num_conversion = {}
for word in rand :
	num_conversion[rand[word]] = words[word]
app = parse_file('/fs3/home/enagaraj/project/test_files/768.statement_of_purpose.Eela_Nagaraj.txt', words, test_seed)
short_app = parse_file('/fs3/home/enagaraj/project/test_files/short_statement.txt', words, test_seed)

unrandomized = []
for num in short_app :
	unrandomized.append(num_conversion[num])

unrandomized_long = []
for n in app :
	unrandomized_long.append(num_conversion[n])


# double check result against tokenizer outputs --> currently works

with open('/fs3/home/enagaraj/project/test_files/short_statement.txt', 'r') as check :
	cleaned = clean_text(check.read())

tokenizer = text.Tokenizer()
tokenizer.word_index = words
txt = [str((" ".join(cleaned)))]
seq = tokenizer.texts_to_sequences(txt)

print [unrandomized] == seq


""" process first num_files files in a directory and output list of strings
	and list of labels with corresponding indices  """
def process_text_files(file_dir, num_files) :
	strings = [0]*num_files
	labels = [0]*num_files
	for filename in os.listdir(file_dir) :
		# index and rating contained in filename
		filepath = os.path.join(file_dir, filename)
		without_ext = filename.split('.')
		index, rating = without_ext[0].split('_')
		# only include files less than this num 
		index = int(index)
		rating = int(rating)
		if index < num_files :
			f = open(filepath, 'r')
			cleaned = clean_review(f.read())	
			strings[index] = cleaned
			labels[index] = rating
			f.close()
	return strings, labels


