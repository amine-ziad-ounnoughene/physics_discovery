import numpy as np
import torch

def target_loss(pred,answer):
	"""
	
	"""
	pred = pred[:,0]
	
	return torch.mean(torch.sum((pred - answer)**2), dim=0)