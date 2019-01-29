from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 
from torch.autograd import Variable

class NCF(nn.Module):
	def __init__(self, user_size, item_size, embed_size, dropout):
		super(NCF, self).__init__()
		self.user_size = user_size
		self.item_size = item_size
		self.embed_size = embed_size
		self.dropout = dropout
		self.a=0.2   #拼接系数
		# Custom weights initialization.
		def init_weights(m):
			# if isinstance(m, nn.Conv2d):
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias, 0)

		self.embed_user_GMF = nn.Linear(self.user_size, self.embed_size)
		self.embed_user_MLP = nn.Linear(self.user_size, self.embed_size)
		self.embed_item_GMF = nn.Linear(self.item_size, self.embed_size)
		self.embed_item_MLP = nn.Linear(self.item_size, self.embed_size)

		# self.embed_user_GMF.apply(init_weights)
		self.embed_user_MLP.apply(init_weights)
		# self.embed_item_GMF.apply(init_weights)
		self.embed_item_MLP.apply(init_weights)

		self.MLP_layers = nn.Sequential(
			nn.Linear(embed_size*2, embed_size*2),
			nn.ReLU(),
			nn.Dropout(p=self.dropout),
			nn.Linear(embed_size*2, embed_size),
			nn.ReLU(),
			nn.Dropout(p=self.dropout),
			nn.Linear(embed_size, 1),
			)
		self.MLP_layers.apply(init_weights)

# 		self.predict_layer = nn.Linear(embed_size*3//2, 1)
# 		self.predict_layer.apply(init_weights)

	def convert_one_hot(self, feature, size):
		""" Convert user and item ids into one-hot format. """
		batch_size = feature.shape[0]
		feature = feature.view(batch_size, 1)
		f_onehot = torch.cuda.FloatTensor(batch_size, size)

		f_onehot.zero_()
		f_onehot.scatter_(-1, feature, 1)

		return f_onehot

	def forward(self, user, item,pre1,pre2,pre3,pre4,pre5):
		user = self.convert_one_hot(user, self.user_size)
   
		pre1=  self.convert_one_hot(pre1, self.item_size)
		pre2=  self.convert_one_hot(pre2, self.item_size)
		pre3=  self.convert_one_hot(pre3, self.item_size)
		pre4=  self.convert_one_hot(pre4, self.item_size)
		pre5=  self.convert_one_hot(pre5, self.item_size)
		item = self.convert_one_hot(item, self.item_size)


		embed_user_MLP = F.relu(self.embed_user_MLP(user))

		embed_item_MLP = F.relu(self.embed_item_MLP(item))
        
		embed_item_pre1 = F.relu(self.embed_item_MLP(pre1))
		embed_item_pre2 = F.relu(self.embed_item_MLP(pre2))
		embed_item_pre3 = F.relu(self.embed_item_MLP(pre3))
		embed_item_pre4 = F.relu(self.embed_item_MLP(pre4))
		embed_item_pre5 = F.relu(self.embed_item_MLP(pre5))
       

		wi1=torch.sum(embed_item_MLP*embed_item_pre1,1)
		wi2=torch.sum(embed_item_MLP*embed_item_pre2,1)           
		wi3=torch.sum(embed_item_MLP*embed_item_pre3,1)        
		wi4=torch.sum(embed_item_MLP*embed_item_pre4,1)
		wi5=torch.sum(embed_item_MLP*embed_item_pre5,1)
         
		summ=torch.exp(wi1)+torch.exp(wi2)+torch.exp(wi3)+torch.exp(wi4)+torch.exp(wi5)
		zi1= torch.exp(wi1)/summ
		zi2= torch.exp(wi2)/summ   
		zi3= torch.exp(wi3)/summ
		zi4= torch.exp(wi4)/summ 
		zi5= torch.exp(wi5)/summ

		zi1=zi1.view(-1,1)
		zi2=zi2.view(-1,1)
		zi3=zi3.view(-1,1)
		zi4=zi4.view(-1,1)
		zi5=zi5.view(-1,1)
		hh1=torch.mul(zi1,embed_item_pre1)
		hh2=torch.mul(zi2,embed_item_pre2)
		hh3=torch.mul(zi3,embed_item_pre3)
		hh4=torch.mul(zi4,embed_item_pre4)
		hh5=torch.mul(zi5,embed_item_pre5)
		pum=hh1+hh2+hh3+hh4+hh5
    
        
		pu=embed_user_MLP+self.a*pum
        
		interaction = torch.cat((pu, embed_item_MLP), -1)
		out=self.MLP_layers(interaction)
        
#		out= torch.sum(pu*embed_item_MLP,1)        
        
		prediction = torch.sigmoid(out)

		return prediction.view(-1)


