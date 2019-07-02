import data_handler as dh
import gensim
from gensim.models.keyedvectors import KeyedVectors
import warnings
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import sys
import numpy as np
import math


def remove_words(list1):
	list2 = []
	for i in range(len(list1)):
		if(len(list1[i]) < 2):
			continue
		elif (list1[i][0] == '@'):
			continue
		elif ((list1[i][0] == 'e') and (list1[i][1] == '_')):
			continue			
		else:
			list2.append(list1[i])
					

	return list2

def vectorize(list1, model):
	list2 = []
	for i in range(len(list1)):
		try:
			list2.append(model[list1[i]])
		except KeyError:
			continue
	return list2		
		
def get_un_features(vec_list):
	maximum = []
	minimum = []
	for i in range(len(vec_list)):
		maxi = np.dot(vec_list[i],vec_list[0])/(math.sqrt(np.dot(vec_list[i],vec_list[i])*np.dot(vec_list[0],vec_list[0])))
		mini = np.dot(vec_list[i],vec_list[0])/(math.sqrt(np.dot(vec_list[i],vec_list[i])*np.dot(vec_list[0],vec_list[0])))
		if(i==0):
			maxi = 0.00
		for j in range(len(vec_list)):
			temp = np.dot(vec_list[i],vec_list[j])
			temp1 = np.dot(vec_list[i],vec_list[i])
			temp2 = np.dot(vec_list[j],vec_list[j])
			temp2 = math.sqrt(temp1*temp2)
			temp = temp / temp2
			if((temp < 1.00) and (i != j)):
				if(temp > maxi):
					maxi = temp
				if(temp < mini):
					mini = temp	
		maximum.append(maxi)
		minimum.append(mini)
	for i in range(len(maximum)):
		max_sim_score = max(maximum)
		min_sim_score = min(maximum)				
		# print(max(maximum))
		max_d_score = max(minimum)
		min_d_score = min(minimum)

	return max_sim_score, min_sim_score, max_d_score, min_d_score	

def get_unweighted_features(d, model):
	unweighted_features = np.zeros((len(d),4))

	for i in range(len(d)):
		d_temp = dh.stopword_removal(d[i][2])
		# print(d)
		d_temp = remove_words(d_temp)
		# print(d_temp)
		vec_list = vectorize(d_temp, model)
		# print(vec_list[0])
		# print(np.dot(vec_list[0],vec_list[1]))
		max_sim_score, min_sim_score, max_d_score, min_d_score = get_un_features(vec_list)
		# print(max_sim_score)
		# print(min_sim_score)
		# print(max_d_score)
		# print(min_d_score)
		unweighted_features[i][0] = max_sim_score
		unweighted_features[i][1] = min_sim_score
		unweighted_features[i][2] = max_d_score
		unweighted_features[i][3] = min_d_score
	return unweighted_features	


def get_wt_features(vec_list):
	maximum = []
	minimum = []
	for i in range(len(vec_list)):
		maxi = np.dot(vec_list[i],vec_list[0])/(math.sqrt(np.dot(vec_list[i],vec_list[i])*np.dot(vec_list[0],vec_list[0])))
		mini = np.dot(vec_list[i],vec_list[0])/(math.sqrt(np.dot(vec_list[i],vec_list[i])*np.dot(vec_list[0],vec_list[0])))
		if(i==0):
			maxi = 0.00
			mini = 1.00
		else:
			maxi = maxi/(i*i)
			mini = mini/(i*i)	
		for j in range(len(vec_list)):
			temp = np.dot(vec_list[i],vec_list[j])
			temp1 = np.dot(vec_list[i],vec_list[i])
			temp2 = np.dot(vec_list[j],vec_list[j])
			temp2 = math.sqrt(temp1*temp2)
			temp = temp / temp2
			if((temp < 1.00) and (i != j)):
				temp = temp/((i-j)*(i-j))
				if(temp > maxi):
					maxi = temp
				if(temp < mini):
					mini = temp	
		maximum.append(maxi)
		minimum.append(mini)
	for i in range(len(maximum)):
		max_sim_score = max(maximum)
		min_sim_score = min(maximum)				
		# print(max(maximum))
		max_d_score = max(minimum)
		min_d_score = min(minimum)

	return max_sim_score, min_sim_score, max_d_score, min_d_score	



def get_weighted_features(d, model):
	weighted_features = np.zeros((len(d),4))

	for i in range(len(d)):
		d_temp = dh.stopword_removal(d[i][2])
		# print(d)
		d_temp = remove_words(d_temp)
		# print(d_temp)
		vec_list = vectorize(d_temp, model)
		# print(vec_list[0])
		# print(np.dot(vec_list[0],vec_list[1]))
		max_sim_score, min_sim_score, max_d_score, min_d_score = get_wt_features(vec_list)
		# print(max_sim_score)
		# print(min_sim_score)
		# print(max_d_score)
		# print(min_d_score)
		weighted_features[i][0] = max_sim_score
		weighted_features[i][1] = min_sim_score
		weighted_features[i][2] = max_d_score
		weighted_features[i][3] = min_d_score
	return unweighted_features	



warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

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
# glove2word2vec(glove_input_file=basepath+"/datahandling/glove.6B.300d.txt", word2vec_output_file=basepath+"/datahandling/gensim_glove_vectors.txt")
# model_g = KeyedVectors.load_word2vec_format(basepath+'/datahandling/gensim_glove_vectors.txt', binary=False)

# print('Loading Training Data...')
# d=dh.loaddata(train_file, word_file_path, split_word_path, emoji_file_path)
# print('Training data loading finished...')

# d[index][1] = label, d[index][2] = largetext in list form

# features = get_unweighted_features(d,model)
# features = get_weighted_features(d, model)
