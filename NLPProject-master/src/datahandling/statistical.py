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
import word_embedding as we
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.utils import shuffle


# ---------------------------------------------------------------------------------

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

def pad(X, maxi):
	if(len(X) == maxi):
		return
	else:
		return np.concatenate([X, np.zeros(maxi-len(X))])	


def naive_bayes(X_train,Y_train,X_test,Y_test):
	# classifier = MultinomialNB()
	classifier = BernoulliNB()
	classifier.fit(X_train, Y_train)
	print("accuracy score of naive bayes")
	print(classifier.score(X_test,Y_test))
	filename = './naive_bayes_glove.sav'
	pickle.dump(classifier, open(filename, 'wb'))


def log_regression(X_train,Y_train,X_test,Y_test):
	classifier = linear_model.LogisticRegression()
	classifier.fit(X_train, Y_train)
	print("accuracy of logistic regression")
	print(classifier.score(X_test,Y_test))
	filename = './log_regression_glove.sav'
	pickle.dump(classifier, open(filename, 'wb'))

def svm(X_train,Y_train,X_test,Y_test):
	classifier = LinearSVC()
	classifier.fit(X_train, Y_train)
	print("accuracy of SVM")
	print(classifier.score(X_test,Y_test))
	filename = './svm_glove.sav'
	pickle.dump(classifier, open(filename, 'wb'))

def feed_forward_nn(X_train,Y_train,X_test,Y_test):
	classifier = MLPClassifier(activation = 'tanh', solver='adam', alpha=1e-3,hidden_layer_sizes=(84, 10), random_state=1)
	classifier.fit(X_train, Y_train)
	print("accuracy of feed forward neural network")
	print(classifier.score(X_test,Y_test))	
	filename = './feed_forward_nn_glove.sav'
	pickle.dump(classifier, open(filename, 'wb'))





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

print("loading glove model")
model = gensim.models.KeyedVectors.load_word2vec_format('./glove.840B.300d.w2vformat.txt', binary=False)
print("loaded glove model")


# model2 = KeyedVectors.load_word2vec_format(basepath+'/datahandling/GoogleNews-vectors-negative300.bin', binary=True)

print('Loading Training Data...')
d=dh.loaddata(train_file, word_file_path, split_word_path, emoji_file_path)
print('Training data loading finished...')

# d[index][1] = label, d[index][2] = largetext in list form

# features = get_unweighted_features(d,model)
# features = get_weighted_features(d, model)
X = []
Y = []

print("constructing feature vectors")

maxi = 0
for i in range(len(d)):
	d_temp = dh.stopword_removal(d[i][2])
	d_temp = we.remove_words(d_temp)
	d_temp = vectorize(d_temp, model)
	nlist1 = ngrams(d_temp, 1)
	nlist1 = list(nlist1)
	nlist2 = list(ngrams(d_temp, 2))
	nlist3 = list(ngrams(d_temp, 3))
	feat1 = construct_feat(nlist1, model)
	feat2 = construct_feat(nlist2, model)
	feat3 = construct_feat(nlist3, model)
	try:
		feat = np.concatenate([feat1, feat2, feat3])
	except:
		continue	
	X.append(feat[0])
	for j in range(1,len(feat)):
		X[-1] = np.concatenate([X[-1], feat[j]])
	Y.append(int(d[i][1]))
	maxi = max(maxi,len(X[-1]))
	

for i in range(len(X)):
	X[i] = pad(X[i], maxi)

print("constructed feature vectors")	

X = np.array(X)
Y = np.array(Y)
X,Y = shuffle(X,Y)
Xtr = X[:5000]
Ytr = Y[:5000]
Xts = X[5000:5500]
Yts = Y[5000:5500]

print("brgining training")
# naive_bayes(Xtr, Ytr, Xts, Yts)
svm(Xtr, Ytr, Xts, Yts)
# log_regression(Xtr, Ytr, Xts, Yts)
# feed_forward_nn(Xtr, Ytr, Xts, Yts)
