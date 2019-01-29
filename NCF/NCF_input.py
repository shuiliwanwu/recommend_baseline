from __future__ import absolute_import
from __future__ import division

import numpy as np 
import pandas as pd 

from torch.utils.data import Dataset


# =============================================================================
# DATA_DIR = '/home/yang/Datasets/ml-1m/ratings.dat' # raw data
# DATA_PATH = './Data' # for saving the processed data
# =============================================================================
COLUMN_NAMES = ['user', 'item']


def re_index(s):
	""" for reindexing the item set. """
	i = 0
	s_map = {}
	for key in s:
		s_map[key] = i
		i += 1

	return s_map

def load_data():
    
	train_data=pd.read_csv('train.tsv',header=None,sep='\t',names=COLUMN_NAMES,engine='python')
	print('----------------------------ok')
	test_data=pd.read_csv('test1.tsv',header=None,sep='\t',names=COLUMN_NAMES,engine='python')
	train_data['user']=train_data['user'].astype('int')
	test_data['user']=test_data['user'].astype('int')
    #Forcing the user index begining from 0.
	train_data.user = train_data.user - 1
	test_data.user = test_data.user - 1
    
	train_data.item = train_data.item - 1
	test_data.item =test_data.item - 1
    
	user_set = set(train_data.user.unique())
	item_set = set(train_data.item.unique())
	user_size = len(user_set)
	item_size = len(item_set)

# item_map = re_index(item_set)

	full_data=pd.concat([train_data,test_data])
	full_data=full_data.reset_index(drop=True)
	#Group each user's interactions(purchased items) into dictionary.
	user_bought = {}
	for i in range(len(full_data)):
		u = full_data['user'][i]
		t = full_data['item'][i]
		if u not in user_bought:
			user_bought[u] = []
		user_bought[u].append(t)
	user_negative=np.load('negSamples.npy')

	labels = np.ones(len(train_data), dtype=np.int32)

	train_features = train_data
	train_labels = labels.tolist()
	test_features = test_data
	test_labels = test_data['item'].tolist() #take the groundtruth item as test labels.

	return ((train_features, train_labels), 
			(test_features, test_labels), 
			(user_size, item_size), 
			(user_set, item_set), 
			(user_bought, user_negative))

def add_negative(features, labels, user_negative, numbers, is_training):
	""" Adding negative samples to training and testing data. """
	feature_user, feature_item, labels_add, features_dict = [], [], [], {}

	for i in range(len(features)):
		user = features['user'][i]
		item = features['item'][i]
		label = labels[i]

		feature_user.append(user)
		feature_item.append(item)
		labels_add.append(label)

		#uniformly sample negative ones from candidate negative items
		neg_samples = np.random.choice(user_negative[user], size=numbers, 
								replace=False).tolist()

		if is_training:
			for k in neg_samples:
				feature_user.append(user)
				feature_item.append(k)
				labels_add.append(0)
		else:
			for k in neg_samples:
				feature_user.append(user)
				feature_item.append(k)
				labels_add.append(k)

	features_dict['user'] = feature_user
	features_dict['item'] = feature_item

	return features_dict, labels_add


class NCFDataset(Dataset):
	def __init__(self, features, labels):
		"""
		After load_data processing, read train or test data. Num_neg is different for
		train and test. User_neg is the items that users have no explicit interaction.
		"""
		self.features = features
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		user = self.features['user'][idx]
		item = self.features['item'][idx]
		label = self.labels[idx]

		sample = {'user': user, 'item': item, 'label': label}

		return sample
