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
import nltk.data

from keras.preprocessing import sequence, text
from keras.utils import np_utils
from keras.datasets import imdb
from bs4 import BeautifulSoup

txt = "that is so readily applicable across academic disciplines.  Since<bear> dAWG I am especially 12-14 intrigued by the intersection between Computer Science and Neuroscience, 1st I am pursuing the Mind, Brain, Behavior focus track within Harvard's Computer Science concentration."
sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')

""" take in raw text, converts each sentence to list of words 
	without html markup, etc, and returns list of sentence lists."""
# pass in text detector so don't keep reloading for each file
def clean_text (raw_text, detector) :
    text = BeautifulSoup(raw_text, "html.parser").get_text()    
    sentences = detector.tokenize(text.strip())
    # go through and convert each sentence to a list of lowercase tokens
    sentence_list = []
    for sentence in sentences :
		text = sentence.lower()
		word_lst = re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", text)
		sentence_list.append(word_lst)
    return sentence_list

pattern = r'''(?x)    # set flag to allow verbose regexps
...     ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
...   | \w+(-\w+)*        # words with optional internal hyphens
...   | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
...   | \.\.\.            # ellipsis
...   | [][.,;"'?():-_`]  # these are separate tokens; includes ],
'''
a = nltk.regexp_tokenize(txt, pattern)


# test for sentence functionality
cleaned =clean_text(txt, sentence_detector)


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
def parse_file (file_path, word_dict, detector, seed) :
	# read in file and get rid of markup
	with open(file_path, 'r') as f :
		cleaned_sentences = clean_text(f.read(), detector)
	# randomize dictionary according to seed
	np.random.seed(seed)
	#rand_dict = dict(
	#	zip(word_dict.keys(), np.random.permutation(word_dict.values())))
	
	# for testing
	rand_dict = word_dict	
	# convert each sentence to a sequence
	sequence_list = []
	for sentence in cleaned_sentences :
		sequence = []
		for word in sentence :
			# if word not in dictionary don't include
			if word in rand_dict :
				sequence.append(rand_dict[word])
		sequence_list.append(sequence)
	return sequence_list



# testing --> convert randomized seq to the same seq outputted by tokenizer

# fix random seed generator and test to make sure still works
test_seed = 19
rand = randomize_dict(words, test_seed)
num_conversion = {}
for word in rand :
	num_conversion[rand[word]] = words[word]
app = parse_file('/fs3/home/enagaraj/project/test_files/768.statement_of_purpose.Eela_Nagaraj.txt', words, sentence_detector, test_seed)
short_app = parse_file('/fs3/home/enagaraj/project/test_files/short_statement.txt', words, sentence_detector, test_seed)

"""
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

"""
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


