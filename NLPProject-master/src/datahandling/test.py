import os
import sys
import numpy as np
import re
import data_handler as dh
import gensim
from gensim.models.keyedvectors import KeyedVectors
import warnings
from gensim.scripts.glove2word2vec import glove2word2vec
import math
# import word_embedding as we
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from collections import Counter

def construct_feat(nlist, model):
	if(len(nlist)!=0):
		feature = np.zeros((len(nlist),900))
		if(len(nlist[0])==1):
			for i in range(len(nlist)):
				x = model[nlist[i][0]]
				y = np.zeros(300)
				z = np.zeros(300)
				t = np.concatenate([x,y,z])
				feature[i] = t
		if(len(nlist[0])==2):
			for i in range(len(nlist)):
				x = model[nlist[i][0]]
				y = model[nlist[i][1]]
				z = np.zeros(300)
				t = np.concatenate([x,y,z])
				feature[i] = t		
		if(len(nlist[0])==3):
			for i in range(len(nlist)):
				x = model[nlist[i][0]]
				y = model[nlist[i][1]]
				z = model[nlist[i][2]]
				t = np.concatenate([x,y,z])
				feature[i] = t
		return feature
	return					


def vectorize(list1, model):
	list2 = []
	list3 = []
	for i in range(len(list1)):
		try:
			list2.append(model[list1[i]])
			list3.append(list1[i])
		except KeyError:
			continue
	return list3


basepath = os.getcwd()[:os.getcwd().rfind('/')]
train_file = basepath + '/../datasets/train/Train_v1.txt'
validation_file = basepath + '/../datasets/Dev_v1.txt'
test_file = basepath + '/../datasets/test/Test_v1.txt'
word_file_path = basepath + '/../datasets/word_list_freq.txt'
split_word_path = basepath + '/../datasets/word_split.txt'
emoji_file_path = basepath + '/../datasets/emoji_unicode_names_final.txt'
output_file = basepath + '/../datasets/TestResults.txt'
model_file = basepath + '/../datasets/'
vocab_file_path = basepath + '/../datasets/vocab_list.txt'
model = KeyedVectors.load_word2vec_format(basepath+'/datahandling/GoogleNews-vectors-negative300.bin', binary=True)

d_temp = ['never', 'voice', 'protest', 'fed', 'shit', 'digest', 'wish', 'reason', 'flaws', 'open', 'season']
d_temp = vectorize(d_temp, model)
nlist1 = ngrams(d_temp, 1)
nlist1 = list(nlist1)
nlist2 = list(ngrams(d_temp, 2))
nlist3 = list(ngrams(d_temp, 3))
feat1 = construct_feat(nlist1, model)
feat2 = construct_feat(nlist2, model)
feat3 = construct_feat(nlist3, model)
feat = np.concatenate([feat1, feat2, feat3])
print(feat)
print(len(feat))
print(len(feat[0]))