"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

import pdb

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, eps=1e-6):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    # inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss

    
def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    N, T = inputs.shape

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / T


def batch_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, gamma: float, eps=1e-6):
    N, T = inputs.shape
    M, T = targets.shape
    
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    ) # (N,T)
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    ) # (N,T)
    
    pos_bce_loss = torch.einsum("nc,mc->nmc", pos, targets) # (N,M,T)
    neg_bce_loss = torch.einsum("nc,mc->nmc", neg, (1 - targets)) # (N,M,T)
    bce_loss = pos_bce_loss + neg_bce_loss
    
    probs = torch.sigmoid(inputs)
    probs_exp = probs.unsqueeze(1)  # (N,1,T)
    targets_exp = targets.unsqueeze(0)  # (1,M,T)
    
    p_t = probs_exp * targets_exp + (1 - probs_exp) * (1 - targets_exp)  # (N,M,T): To preserve the time-axis focal weight
    focal_weight = ((1 - p_t) + eps) ** gamma
    
    bce_exp = pos_bce_loss * targets_exp + neg_bce_loss * (1 - targets_exp)  # (N, M, T)
    focal_loss = focal_weight * bce_exp # (N,M,T)
    
    weights = weights.view(1, M, 1)
    weights = weights / weights.sum()
    
    weighted_loss = (weights * focal_loss).sum(dim=-1) # (N, M)
    
    # pdb.set_trace()
    
    return weighted_loss / T

def batch_focal_tversky_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, gamma: float, eps=1e-6):
    N, T = inputs.shape
    M, T = targets.shape
    
    inputs = inputs.sigmoid()

    tp = torch.einsum("nc,mc->nm", inputs, targets)  # (N,M)
    fp = torch.einsum("nc,mc->nm", inputs, 1 - targets)  # (N,M)
    fn = torch.einsum("nc,mc->nm", 1 - inputs, targets)  # (N,M)
    
    alpha = alpha.view(1, M)
    beta = beta.view(1, M)
    weights = weights.view(1, M)
    weights = weights / weights.sum()
    
    tversky_index = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    tversky_loss = 1 -  tversky_index
    
    weighted_loss = (tversky_loss + eps) ** gamma
    
    # pdb.set_trace()
    
    return weighted_loss
    
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_query, weight, alpha, gamma, cost_dia, cost_dice, cost_cls):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        weight_expanded = torch.full((num_query,), weight[-1])
        weight_expanded[:len(weight)] = torch.tensor(weight, dtype=torch.float32)
        alpha_expanded = torch.full((num_query,), alpha[-1])
        alpha_expanded[:len(alpha)] = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("weight", weight_expanded)
        self.register_buffer("alpha", alpha_expanded)
        self.register_buffer("beta", 1 - alpha_expanded)
        self.gamma = gamma
        self.cost_cls = cost_cls
        self.cost_dia = cost_dia
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, preds_mask, preds_class, targets):
        """More memory-friendly matching"""
        
        B, T, N = preds_mask.shape
        orig_device = preds_mask.device

        indices = []
        # preds_mask_sorted = []
        # preds_class_sorted = []

        # Iterate through batch size
        for b in range(B):
            
            out_mask = preds_mask[b] # [T N]
            _, N = out_mask.shape
            tgt_mask = targets[b] # [T S]
            # print(tgt_mask, tgt_mask.shape)
            _, L = tgt_mask.shape
            
            # cost_dia = batch_sigmoid_ce_loss(out_mask.T, tgt_mask.T) # [N S]
            # cost_dice = batch_dice_loss(out_mask.T, tgt_mask.T) # [N S]
            cost_dia = batch_focal_loss(out_mask.T, tgt_mask.T, self.weight[:L], 1 - self.gamma)
            cost_dice = batch_focal_tversky_loss(out_mask.T, tgt_mask.T, self.weight[:L], self.alpha[:L], self.beta[:L], self.gamma)
            
            out_prob = preds_class[b].sigmoid() # [N 1]
            cost_cls = -out_prob.repeat(1, L) # [N S]

            costs = [cost_dia, cost_dice, cost_cls]
            # Final cost matrix
            C = (
                self.cost_dia * cost_dia
                + self.cost_dice * cost_dice
                + self.cost_cls * cost_cls
            )
            
            lang_ind, query_ind = linear_sum_assignment(C.T.detach().cpu())
            lang_ind, query_ind = torch.as_tensor(lang_ind, dtype=torch.int64, device=orig_device), torch.as_tensor(query_ind, dtype=torch.int64, device=orig_device)
            indices.append((lang_ind, query_ind))
            
            # query_ind_all = torch.arange(N).to(orig_device)
            # rest_ind = query_ind_all[~torch.isin(query_ind_all, query_ind)]
            
            # # print(out_mask.device, out_prob.device)
            # sorted_mask = out_mask[:, query_ind]
            # sorted_class = torch.cat([out_prob.squeeze()[query_ind], out_prob.squeeze()[rest_ind]], dim=-1)
            # # print(sorted_mask.device, sorted_class.device)
            
            # preds_mask_sorted.append(sorted_mask)
            # preds_class_sorted.append(sorted_class)

        return costs, indices

# class HungarianMatcher(nn.Module):
#     """This class computes an assignment between the targets and the predictions of the network

#     For efficiency reasons, the targets don't include the no_object. Because of this, in general,
#     there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
#     while the others are un-matched (and thus treated as non-objects).
#     """

#     def __init__(self, cost_dia: float = 5, cost_dice: float = 5, cost_class: float = 2):
#         """Creates the matcher

#         Params:
#             cost_class: This is the relative weight of the classification error in the matching cost
#             cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
#             cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
#         """
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_dia = cost_dia
#         self.cost_dice = cost_dice

#         assert cost_class != 0 or cost_dia != 0 or cost_dice != 0, "all costs cant be 0"

#     @torch.no_grad()
#     def forward(self, preds_mask, preds_class, targets):
#         """More memory-friendly matching"""
#         B, T, N = preds_mask.shape
#         B_, T_, S = targets.shape
        
#         assert B == B_ and T == T_

#         indices = []

#         # Iterate through batch size
#         for b in range(B):
            
#             out_mask = preds_mask[b] # [T N]
#             tgt_mask = targets[b] # [T S]
#             _, S = tgt_mask.shape
            
#             cost_dia = batch_sigmoid_ce_loss(out_mask.T, tgt_mask.T) # [N S]
#             cost_dice = batch_dice_loss(out_mask.T, tgt_mask.T) # [N S]
            
#             # 이부분 확실치 않음
#             out_prob = preds_class[b].softmax(dim=-2) # [N]
#             cost_class = -out_prob.repeat(1, S) # [N S]

#             # Final cost matrix
#             C = (
#                 self.cost_dia * cost_dia
#                 + self.cost_dice * cost_dice
#                 + self.cost_class * cost_class
#             )

#             indices.append(linear_sum_assignment(C))

#         return [
#             (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#             for i, j in indices
#         ] # From query idx to language idx