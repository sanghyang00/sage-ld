import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

import pdb
    
# 0: Mute / 1: Matrix language (major) / 2~: Embedded language (Minor)
class FocalLoss(nn.Module): # If weights are all identical and gamma = 0, same with weighted BCE
    def __init__(self, num_query, weight_mute, weight_matrix, weight_embedded, gamma):
        super().__init__()
        weight = torch.ones(num_query)
        weight[0] = weight_mute
        weight[1] = weight_matrix
        weight[2:] = weight_embedded
        self.register_buffer("weight", weight)
        self.gamma = 1 - gamma
        
    def forward(self, pred, target, mask=None):
        
        assert pred.shape == target.shape
        T, L = pred.shape
        pred = pred.sigmoid().clamp(min=EPS, max=1-EPS)
        
        if mask is not None:
            pred = pred * mask.unsqueeze(1)
            target = target * mask.unsqueeze(1)
        
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        pred_t = pred * target + (1 - pred) * (1 - target)
        
        alpha = self.weight[:L]
        alpha = alpha / alpha.sum() # normalize weight
        alpha = alpha.view(1, L)
        
        focal_loss = (alpha * bce_loss * (((1 - pred_t) + EPS) ** self.gamma)).mean()
        
        # pdb.set_trace()
        
        return focal_loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, num_query, alpha_mute, alpha_matrix, alpha_embedded, gamma):
        super().__init__()
        alpha = torch.ones(num_query)
        alpha[0] = alpha_mute
        alpha[1] = alpha_matrix
        alpha[2:] = alpha_embedded
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", 1 - alpha)
        self.gamma = gamma
        
    def forward(self, pred, target, mask=None):
        
        assert pred.shape == target.shape
        
        pred = pred.sigmoid().clamp(min=EPS, max=1-EPS) # logit to probability
        T, L = pred.shape
        
        if mask is not None:
            pred = pred * mask.unsqueeze(1)
            target = target * mask.unsqueeze(1)
        
        alpha = self.alpha[:L].view(1,L)
        beta = self.beta[:L].view(1,L)
        
        tp = (pred * target).sum(dim=0)
        fp = (pred * (1 - target)).sum(dim=0)
        fn = ((1 - pred) * target).sum(dim=0)
        
        tversky_index = (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)
        tversky_loss = 1 - tversky_index
        
        weighted_loss = ((tversky_loss + EPS) ** self.gamma).mean()
        
        # pdb.set_trace()
        
        return weighted_loss
    
# Deprecated due to the unstable learning
class WeightedFocalTverskyLoss(nn.Module):
    def __init__(self, num_query, weight_mute, weight_matrix, weight_embedded, alpha_mute, alpha_matrix, alpha_embedded, gamma):
        super().__init__()
        weight = torch.ones(num_query)
        weight[0] = weight_mute
        weight[1] = weight_matrix
        weight[2:] = weight_embedded
        alpha = torch.ones(num_query)
        alpha[0] = alpha_mute
        alpha[1] = alpha_matrix
        alpha[2:] = alpha_embedded
        self.register_buffer("weight", weight)
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", 1 - alpha)
        self.gamma = gamma
        
    def forward(self, pred, target, mask=None):
        
        assert pred.shape == target.shape
        
        pred = pred.sigmoid().clamp(min=EPS, max=1-EPS) # logit to probability
        T, L = pred.shape
        
        if mask is not None:
            pred = pred * mask.unsqueeze(1)
            target = target * mask.unsqueeze(1)
        
        weights = self.weight[:L]
        weights = weights / weights.sum() # normalize weight
        weights = weights.view(1, L)
        
        alpha = self.alpha[:L].view(1,L)
        beta = self.beta[:L].view(1,L)
        
        tp = (pred * target).sum(dim=0)
        fp = (pred * (1 - target)).sum(dim=0)
        fn = ((1 - pred) * target).sum(dim=0)
        
        tversky_index = (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)
        tversky_loss = 1 - tversky_index
        
        weighted_loss = (weights * (tversky_loss ** self.gamma)).sum() # / weights.sum()
        
        # pdb.set_trace()
        
        return weighted_loss
    
class ClassificationLoss(nn.Module):
    def __init__(self, neg_cls_factor=0.2):
        super().__init__()
        self.neg_cls_factor = neg_cls_factor
        
    def forward(self, pred, target):
        
        pred = pred.sigmoid().clamp(min=EPS, max=1-EPS)
        N = pred.shape
        T, L = target.shape
        ref = torch.zeros(N, device=pred.device)
        ref[:L] = 1
        weight = torch.ones(N, device=pred.device) * self.neg_cls_factor
        weight[:L] = 1.0
        # print(pred.device, ref.device, weight.device)
            
        cls_loss = F.binary_cross_entropy(pred, ref, weight=weight)
        
        return cls_loss
    
class EELDLoss(nn.Module):
    def __init__(self, num_query, weight, alpha, gamma, dia_weight, dice_weight, cls_weight, neg_cls_factor):
        super().__init__()
        self.dia_loss = FocalLoss(num_query=num_query, weight_mute=weight[0], weight_matrix=weight[1], weight_embedded=weight[2], gamma=gamma)
        self.dia_weight = dia_weight
        self.dice_loss = FocalTverskyLoss(num_query=num_query, alpha_mute=alpha[0], alpha_matrix=alpha[1], alpha_embedded=alpha[2], gamma=gamma)
        self.dice_weight = dice_weight
        
        self.cls_loss = ClassificationLoss(neg_cls_factor=neg_cls_factor)
        self.cls_weight = cls_weight
        
    def forward(self, pred_mask, pred_cls, target, mask=None):
    
        loss_dia = self.dia_weight * self.dia_loss(pred_mask, target, mask=mask)
        loss_dice = self.dice_weight * self.dice_loss(pred_mask, target, mask=mask)
        loss_cls = self.cls_weight * self.cls_loss(pred_cls, target)
        
        total_loss = loss_dia + loss_dice + loss_cls
        
        return total_loss, loss_dia, loss_dice, loss_cls