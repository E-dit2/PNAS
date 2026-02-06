import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn import metrics
from sklearn.metrics import auc
#from torch.autograd.grad_mode import F
# training loss, sigmoid, false negative loss, which should be non-negative


# training loss, sigmoid, oversampled PU
class OversampledPULossFunc(nn.Module):
    def __init__(self):
        super(OversampledPULossFunc, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        return

    def forward(self, output_p, output_n, prior, prior_prime):
        prior_prime=0.5
        labelp=torch.ones(output_p.shape[0],device=output_p.device,dtype=torch.long)
        labeln=torch.zeros(output_n.shape[0],device=output_n.device,dtype=torch.long)
        cost = self.cross_entropy(output_n, labeln) * (1 - (prior_prime - prior) / (1 - prior))
        cost = (1 - prior_prime) / (1 - prior) * cost
        cost = cost + self.cross_entropy(output_p, labelp)
        return cost

class OversampledNNPULossFunc(nn.Module):
    def __init__(self):
        super(OversampledNNPULossFunc, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        return

    def forward(self, output_p, output_n, prior, prior_prime):
        prior_prime=0.5
        labelp=torch.ones(output_p.shape[0],device=output_p.device,dtype=torch.long)
        labeln=torch.zeros(output_n.shape[0],device=output_n.device,dtype=torch.long)
        cost = self.cross_entropy(output_n, labeln) * (1 - (prior_prime - prior) / (1 - prior))
        cost = (1 - prior_prime) / (1 - prior) * cost
        return cost
# training loss, sigmoid, PN
class PNTrainingSigmoid(nn.Module):
    def __init__(self):
        super(PNTrainingSigmoid, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        cost = prior * torch.mean(torch.sigmoid(-output_p))
        cost = cost + (1 - prior) * torch.mean(torch.sigmoid(output_n))
        return cost





# non-negative training loss, 0-1
class NNZeroOneTrain(nn.Module):
    def __init__(self):
        super(NNZeroOneTrain, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        cost = torch.mean((1 + torch.sign(output_n)) / 2) - prior * torch.mean((1 + torch.sign(output_p)) / 2)
        cost = max(cost, 0)
        cost = cost + prior * torch.mean((1 - torch.sign(output_p)) / 2)
        return cost


# test loss, 0-1
class ZeroOneTest(nn.Module):
    def __init__(self):
        super(ZeroOneTest, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        cost = prior * torch.mean((1 - torch.sign(output_p)) / 2)
        cost = cost + (1 - prior) * torch.mean((1 + torch.sign(output_n)) / 2)
        return cost

### precision, recall, F1 and AUC for test
class PUPrecision(nn.Module):
    def __init__(self):
        super(PUPrecision, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        true_positive = torch.sum((torch.sign(output_p) + 1)/2)
        all_predicted_positive = torch.sum((torch.sign(output_p) + 1)/2) + torch.sum((torch.sign(output_n)+1)/2)
        if all_predicted_positive == 0:
            precision = 0
        else:
            precision = float(true_positive)/float(all_predicted_positive)
        return precision

class PURecall(nn.Module):
    def __init__(self):
        super(PURecall, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        true_positive = torch.sum((torch.sign(output_p) + 1) / 2)
        all_real_positive = len(output_p)
        recall = float(true_positive) / float(all_real_positive)
        return recall


class PUF1(nn.Module):
    def __init__(self):
        super(PUF1, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        true_positive = torch.sum((torch.sign(output_p) + 1) / 2)
        all_predicted_positive = torch.sum((torch.sign(output_p) + 1) / 2)+ torch.sum((torch.sign(output_n) + 1) / 2)
        all_real_positive = len(output_p)
        if all_predicted_positive == 0:
            precision = 0
        else:
            precision = float(true_positive) / float(all_predicted_positive)
        recall = float(true_positive) / float(all_real_positive)
        if precision ==0 or recall == 0:
            F1 = 0
        else:
            F1 = 2*precision*recall/(precision+recall)
        return F1



class Auc_loss(nn.Module):
    def __init__(self):
        super(Auc_loss, self).__init__()
        return

    def forward(self, out_put, target):
        fpr, tpr, thresholds = metrics.roc_curve(target.cpu().numpy(),

                                                 out_put.cpu().numpy(), pos_label=1)

        Auc = metrics.auc(fpr, tpr)

        return Auc

class SelfSupConLoss(nn.Module):
	"""
	Self Sup Con Loss: https://arxiv.org/abs/2002.05709
	Adopted from lightly.loss.NTXentLoss :
	https://github.com/lightly-ai/lightly/blob/master/lightly/loss/ntx_ent_loss.py
	"""
	
	def __init__(self, temperature: float = 0.5, reduction="mean"):
		super(SelfSupConLoss, self).__init__()
		self.temperature = temperature
		self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
	
	def forward(self, z: torch.Tensor, z_aug: torch.Tensor, *kwargs) -> torch.Tensor:
		"""
		:param z: features
		:param z_aug: augmentations
		:return: loss value, scalar
		"""
		# get inner product matrix - diag zero
		batch_size, _ = z.shape
		
		# project onto hypersphere
		z = nn.functional.normalize(z, dim=1)
		z_aug = nn.functional.normalize(z_aug, dim=1)
		
		# calculate similarities block-wise - the resulting vectors have shape (batch_size, batch_size)
		inner_pdt_00 = torch.einsum('nc,mc->nm', z, z) / self.temperature
		inner_pdt_01 = torch.einsum('nc,mc->nm', z, z_aug) / self.temperature
		inner_pdt_10 = torch.einsum("nc,mc->nm", z_aug, z) / self.temperature
		inner_pdt_11 = torch.einsum('nc,mc->nm', z_aug, z_aug) / self.temperature
		
		# remove similarities between same views of the same image
		diag_mask = torch.eye(batch_size, device=z.device, dtype=torch.bool)
		inner_pdt_00 = inner_pdt_00[~diag_mask].view(batch_size, -1)
		inner_pdt_11 = inner_pdt_11[~diag_mask].view(batch_size, -1)
		
		# concatenate blocks : o/p shape (2*batch_size, 2*batch_size) - diagonals (self sim) zero
		# [ Block 01 ] | [ Block 00 ]
		# [ Block 10 ] | [ Block 11 ]
		inner_pdt_0100 = torch.cat([inner_pdt_01, inner_pdt_00], dim=1)
		inner_pdt_1011 = torch.cat([inner_pdt_10, inner_pdt_11], dim=1)
		logits = torch.cat([inner_pdt_0100, inner_pdt_1011], dim=0)
		
		labels = torch.arange(batch_size, device=z.device, dtype=torch.long)
		labels = labels.repeat(2)
		loss = self.cross_entropy(logits, labels)
		
		return loss


class SupConLoss(nn.Module):
	"""
	Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	Attractive force between self augmentation and all other samples from same class
	"""
	
	def __init__(self, temperature: float = 0.5, reduction="mean"):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
		self.reduction = reduction
	
	def forward(self, z: torch.Tensor, z_aug: torch.Tensor, labels: torch.Tensor, *kwargs) -> torch.Tensor:
		"""
		
		:param z: features => bs * shape
		:param z_aug: augmentations => bs * shape
		:param labels: ground truth labels of size => bs
		:return: loss value => scalar
		"""
		batch_size, _ = z.shape
		
		# project onto hypersphere
		z = nn.functional.normalize(z, dim=1)
		z_aug = nn.functional.normalize(z_aug, dim=1)
		
		# calculate similarities block-wise - the resulting vectors have shape (batch_size, batch_size)
		inner_pdt_00 = torch.einsum('nc,mc->nm', z, z) / self.temperature
		inner_pdt_01 = torch.einsum('nc,mc->nm', z, z_aug) / self.temperature
		inner_pdt_10 = torch.einsum("nc,mc->nm", z_aug, z) / self.temperature
		inner_pdt_11 = torch.einsum('nc,mc->nm', z_aug, z_aug) / self.temperature
		
		# concatenate blocks : o/p shape (2*batch_size, 2*batch_size) - diagonals (self sim) zero
		# [ Block 00 ] | [ Block 01 ]
		# [ Block 10 ] | [ Block 11 ]
		inner_pdt_0001 = torch.cat([inner_pdt_00, inner_pdt_01], dim=1)
		inner_pdt_1011 = torch.cat([inner_pdt_10, inner_pdt_11], dim=1)
		inner_pdt_mtx = torch.cat([inner_pdt_0001, inner_pdt_1011], dim=0)
		
		max_inner_pdt, _ = torch.max(inner_pdt_mtx, dim=1, keepdim=True)
		inner_pdt_mtx = inner_pdt_mtx - max_inner_pdt.detach()  # for numerical stability
		
		# compute negative log-likelihoods
		nll_mtx = torch.exp(inner_pdt_mtx)
		# mask out self contrast
		diag_mask = torch.ones_like(inner_pdt_mtx, device=z.device, dtype=torch.bool).fill_diagonal_(0)
		nll_mtx = nll_mtx * diag_mask
		nll_mtx /= torch.sum(nll_mtx, dim=1, keepdim=True)
		nll_mtx[nll_mtx != 0] = - torch.log(nll_mtx[nll_mtx != 0])
		
		# mask out contributions from samples not from same class as i
		mask_label = torch.unsqueeze(labels, dim=-1)
		eq_mask = torch.eq(mask_label, torch.t(mask_label))
		eq_mask = torch.tile(eq_mask, (2, 2))
		similarity_scores = nll_mtx * eq_mask
		
		# compute the loss -by averaging over multiple positives
		loss = similarity_scores.sum(dim=1) / (eq_mask.sum(dim=1) - 1)
		if self.reduction == 'mean':
			loss = torch.mean(loss)
		return loss


class PUConLoss(nn.Module):
	"""
    Proposed PUConLoss : leveraging available positives only
    """
	
	def __init__(self, temperature: float = 0.5):
		super(PUConLoss, self).__init__()
		# per sample unsup and sup loss : since reduction is None
		self.sscl = SelfSupConLoss(temperature=temperature, reduction='none')
		self.scl = SupConLoss(temperature=temperature, reduction='none')
	
	def forward(self, z: torch.Tensor, z_aug: torch.Tensor, labels: torch.Tensor, *kwargs) -> torch.Tensor:
		"""
        @param z: Anchor
        @param z_aug: Mirror
        @param labels: annotations
        """
		# get per sample sup and unsup loss
		sup_loss = self.scl(z=z, z_aug=z_aug, labels=labels)
		unsup_loss = self.sscl(z=z, z_aug=z_aug)
		
		# label for M-viewed batch with M=2
		labels = labels.repeat(2).to(z.device)
		
		# get the indices of P and  U samples in the multi-viewed batch
		p_ix = torch.where(labels == 1)[0]
		u_ix = torch.where(labels == 0)[0]
		
		# if no positive labeled it is simply SelfSupConLoss
		num_labeled = len(p_ix)
		if num_labeled == 0:
			return torch.mean(unsup_loss)
		
		# compute expected similarity
		# -------------------------
		risk_p = sup_loss[p_ix]
		risk_u = unsup_loss[u_ix]
		
		loss = torch.cat([risk_p, risk_u], dim=0)
		return torch.mean(loss)