from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import numpy as np

# def ap(gt_item, pred_items):
#     """Compute the average precision (AP) of a list of ranked items
#     """
#     if gt_item in pred_items:
#         return 1
#     else:
#         return 0
# def ap(gt_item, pred_items):
#     """Compute the average precision (AP) of a list of ranked items
#     """
#     hits = 0
#     sum_precs = 0
#     pred_items=pred_items.cpu().data.numpy()
#     for n in range(len(pred_items)):
#         if ranked_list[n] in gt_item:
#             hits += 1
#             sum_precs += hits / (n + 1.0)
#     if hits > 0:
#         return sum_precs / len(gt_item)
#     else:
#         return 0
def ap(gt_item, pred_items):
	if gt_item in pred_items:
		index = torch.tensor(pred_items.tolist().index(gt_item),
							dtype=torch.float32)
		return 1.0/(float(index + 1))
	else:
		return 0
def mrr(gt_item, pred_items):
    if gt_item in pred_items:
        index = np.where(pred_items == gt_item)[0][0]
        return np.reciprocal(float(index + 1))
    else:
        return 0

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = torch.tensor(pred_items.tolist().index(gt_item),
							dtype=torch.float32)
		return torch.reciprocal(torch.log2(index+2))
	return 0

def metrics(model, test_dataloader, top_k):
	AP,MRR,HR, NDCG =[],[], [],[]
	with torch.no_grad():
		for batch_data in test_dataloader:
			user = batch_data['user'].long().cuda()
			item = batch_data['item'].long().cuda()
			pre1  = batch_data['pre1'].long().cuda()
       
			pre2  = batch_data['pre2'].long().cuda()
			pre3  = batch_data['pre3'].long().cuda()
			pre4  = batch_data['pre4'].long().cuda()
			pre5  = batch_data['pre5'].long().cuda()
            # label = batch_data['label'].numpy()

			prediction = model(user, item,pre1,pre2,pre3,pre4,pre5)
			_, indices = torch.topk(prediction, top_k)
			recommend = torch.take(item, indices)
        
			AP.append(ap(item[0], recommend))        
			MRR.append(mrr(item[0], recommend))
			HR.append(hit(item[0], recommend))
			NDCG.append(ndcg(item[0], recommend))

	return np.mean(AP),np.mean(MRR),np.mean(HR), np.mean(NDCG)

