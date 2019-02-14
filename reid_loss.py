from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from mmd import mix_rbf_mmd2
from reid_dataset import split_datapack

class ReconstructionLoss(nn.Module):
    def __init__(self, dist_metric='L1'):
        super(ReconstructionLoss, self).__init__()
        self.dist_metric = dist_metric
        
    def forward(self, re_img, gt_img):
        p = 2
        if self.dist_metric == 'L1':
            p = 1
        b,c,h,w = gt_img.size()
        loss = torch.dist(re_img, gt_img, p=p) / (b*h*w)
        return loss

class MMDLoss(nn.Module):
    def __init__(self, base=1.0, sigma_list=[1, 2, 10]):
        super(MMDLoss, self).__init__()
        # sigma for MMD
        #         self.sigma_list = sigma_list
        self.base = base
        self.sigma_list = sigma_list
        self.sigma_list = [sigma / self.base for sigma in self.sigma_list]

    def forward(self, Target, Source):
        Target = Target.view(Target.size()[0], -1)
        Source = Source.view(Source.size()[0], -1)
        mmd2_D = mix_rbf_mmd2(Target, Source, self.sigma_list)
        mmd2_D = F.relu(mmd2_D)
        mmd2_D = torch.sqrt(mmd2_D)
        return mmd2_D

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, predict, gt):
        predict = predict.view(predict.size()[0], -1)
        batch, dim = predict.size()
        loss = 0.0
        for i in range(batch):
            for j in range(i, batch):
                if gt[i] == gt[j]:
                    label = 1
                else:
                    label = 0
                dist = torch.dist(predict[i], predict[j], p=2) ** 2 / dim
                loss += label * dist + (1 - label) * F.relu(self.margin - dist)
        loss = 2 * loss / (batch * (batch - 1))
        return loss

def loss_ctr_func(pred, gt):
    criterion = ContrastiveLoss().cuda()
    loss = criterion(pred, gt)
    return loss

def loss_rec_func(pred, gt):
    criterion = ReconstructionLoss(dist_metric='L1').cuda()
    loss = criterion(pred, gt)
    return loss

def loss_cls_func(pred_label, gt_label):
    criterion = nn.CrossEntropyLoss().cuda()
    loss = criterion(pred_label, gt_label)
    return loss

def loss_dif_func(feature1, feature2):
    feature1 = feature1.view(-1)
    feature2 = feature2.view(-1)
    loss = torch.dot(feature1, feature2)
    return loss

def loss_mmd_func(target_feature, source_feature):
    # We reserve the rights to provide the optimal sigma list
    criterion = MMDLoss(sigma_list=[1, 2, 10]) 
    loss = criterion(target_feature, source_feature)
    return loss


def loss_triplet(global_feature, local_feature, label, normalize=True):
    criterion = TripletLoss(margin=config.GLOBAL_MARGIN)
    global_loss, pos_inds, neg_inds = GlobalLoss(criterion, 
                                                  global_feature,
                                                  label.cuda(args.gpu),
                                                  normalize_feature=normalize)

    criterion = TripletLoss(margin=config.LOCAL_MARGIN)
    

    return global_loss


"""
New added losses
"""    

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input      
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def batch_euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [N, m, d]
        y: pytorch Variable, with shape [N, n, d]
    Returns:
        dist: pytorch Variable, with shape [N, m, n]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    N, m, d = x.size()
    N, n, d = y.size()

    # shape [N, m, n]
    xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
    yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
    dist = xx + yy
    dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
        dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
        labels: pytorch LongTensor, with shape [N]
        return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        p_inds: pytorch LongTensor, with shape [N]; 
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples, 
        thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
   
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze( 0).expand(N, N))

        # shape [N, 1]
        p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)

        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def GlobalLoss(tri_loss, global_feat, labels, normalize_feature=True):
    """
    Args:
        tri_loss: a `TripletLoss` object
        global_feat: pytorch Variable, shape [N, C]
        labels: pytorch LongTensor, with shape [N]
        normalize_feature: whether to normalize feature to unit length along the 
            Channel dimension
    Returns:
        loss: pytorch Variable, with shape [1]
        p_inds: pytorch LongTensor, with shape [N]; 
            indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
        n_inds: pytorch LongTensor, with shape [N];
            indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        =============
        For Debugging
        =============
        dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
        dist_an: pytorch Variable, distance(anchor, negative); shape [N]
        ===================
        For Mutual Learning
        ===================
        dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    if normalize_feature:
        global_feat = normalize(global_feat, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat, global_feat)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels, return_inds=True)
    loss = tri_loss(dist_ap, dist_an)
    #return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat
    return loss, p_inds, n_inds

    
class TripletLoss(object):
    """
        Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
        Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
        Loss for Person Re-Identification'.
    """
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_ap, dist_an):
        """
            Args:
                dist_ap: pytorch Variable, distance between anchor and positive sample, 
                    shape [N]
                dist_an: pytorch Variable, distance between anchor and negative sample, 
                    shape [N]
            Returns:
                loss: pytorch Variable, with shape [1]
        """
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss