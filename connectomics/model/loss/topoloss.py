from __future__ import print_function, division

import sys
import numpy as np
import torch
import torch.nn.functional as F
from gudhi.wasserstein import wasserstein_distance
import cripser as cr

printstuff = False
class TopoLossMSE3D(torch.nn.Module):
    """3D Topological loss for 3D volumes"""

    def __init__(self, topo_dim, topo_weight):
        super().__init__()
        print("[Saumya] 3D Topo dim: {}; Topo weight: {}".format(topo_dim, topo_weight))
        self.topo_dim = topo_dim # dimension of the topological features (0, 1, or 2 for 3D)
        self.topo_weight = topo_weight # weight of the topological loss

    def forward(self, pred, target, weight_mask=None):
        # pred.size() : [B, C, D, H, W] for 3D volumes
        loss_val = 0.

        B, C, D, H, W = pred.size()

        for b in range(B):  # batch size
            for c in range(C):  # channels
                # Process entire 3D volume at once
                volume_loss = getTopoLoss3d(pred[b, c, :, :, :], target[b, c, :, :, :], self.topo_dim)
                loss_val = loss_val + volume_loss

        loss_val /= (B * C)  # normalize over batch and channels
        loss_val = loss_val * self.topo_weight
        # loss_val *= self.topo_weight
        # loss_val = torch.tensor(loss_val).cuda()

        return loss_val
    
class TopoLossMSE2D_slice(torch.nn.Module):
    """Weighted Topological loss for 3D volumes with 2D slice evaluation"""

    def __init__(self, topo_dim, topo_weight):
        super().__init__()
        print("[Saumya] Topo dim: {}; Topo weight: {}".format(topo_dim, topo_weight))
        self.topo_dim = topo_dim # dimension of the topological features
        self.topo_weight = topo_weight # weight of the topological loss

    def forward(self, pred, target, weight_mask=None):
        # pred.size() : [B, C, D, H, W]
        loss_val = 0.

        B, C, D, H, W = pred.size()

        for b in range(B):  # batch size
            for d in range(D):  # depth slices
                for c in range(C):  # channels
                    slice_loss = getTopoLoss2d(pred[b, c, d, :, :], target[b, c, d, :, :], self.topo_dim)
                    loss_val = loss_val + slice_loss

        loss_val /= (B * C * D)  # normalize over batch, channels, and depth
        loss_val = loss_val * self.topo_weight
        # loss_val = torch.tensor(loss_val).cuda()

        return loss_val
    
class TopoLossMSE2D(torch.nn.Module):
    """Weighted Topological loss
    """

    def __init__(self, topo_dim, topo_weight):
        super().__init__()
        print("[Saumya] Topo dim: {}; Topo weight: {}".format(topo_dim, topo_weight))
        self.topo_dim = topo_dim # dimension of the topological features
        self.topo_weight = topo_weight # weight of the topological loss


    def forward(self, pred, target, weight_mask=None):
        # pred.size() : [8, 3, 64, 64] # NCHW
        loss_val = 0. 

        # print("Saumya:\nPred range [{},{}]\nGT range [{},{}]".format(torch.min(pred),torch.max(pred),torch.min(target),torch.max(target)))

        # print("[Saumya] Size of pred and target: {}, {}".format(pred.size(), target.size()))

        for idx in range(pred.size()[0]): # batchsize=N
            for ch in range(pred.size()[1]): # n_channel=C ; See if we want to perform topoloss on all channels (multi-class), or, only on foreground (binary problem)
                loss_val += getTopoLoss2d(pred[idx, ch, :, : ], target[idx, ch, :, : ], self.topo_dim)
        loss_val /= pred.size()[1] # divide by the number of channels
        loss_val /= pred.size()[0] # divide by the batch size
        loss_val *= self.topo_weight # multiply by the weight of the topological loss
        loss_val = torch.tensor(loss_val).cuda()

        return loss_val 

def compute_dgm_force(stu_lh_dgm, tea_lh_dgm):
    """
    Compute the persistent diagram of the image

    Args:
        stu_lh_dgm: likelihood persistent diagram of student model.
        tea_lh_dgm: likelihood persistent diagram of teacher model.

    Returns:
        idx_holes_to_remove: The index of student persistent points that require to remove for the following training process [aka stu dots matched to diagonal]
        off_diagonal_match: The index pairs of persistent points that requires to fix in the following training process [aka stu dots matched to tea]
    
    """
    if stu_lh_dgm.shape[0] == 0:
        idx_holes_to_remove, off_diagonal_match = np.zeros((0,2)), np.zeros((0,2))
        return idx_holes_to_remove, off_diagonal_match
    
    if (tea_lh_dgm.shape[0] == 0):
        tea_pers = None
        tea_n_holes = 0
    else:
        tea_pers = abs(tea_lh_dgm[:, 1] - tea_lh_dgm[:, 0])
        tea_n_holes = tea_pers.size

    if (tea_pers is None or tea_n_holes == 0):
        idx_holes_to_remove = list(set(range(stu_lh_dgm.shape[0])))
        off_diagonal_match = list()
    else:
        idx_holes_to_remove, off_diagonal_match = get_matchings(stu_lh_dgm, tea_lh_dgm)
    
    return idx_holes_to_remove, off_diagonal_match


def getCriticalPoints_cr(likelihood, topo_dim):
        
    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=1, location="birth") # dim birth death x1  y1  z1  x2  y2  z2
    pd_arr_lh = pd[pd[:, 0] == topo_dim] # 0 or 1-dim topological features
    pd_lh = pd_arr_lh[:, 1:3] # birth time and death time
    # birth critical points
    bcp_lh = pd_arr_lh[:, 3:5]
    # death critical points
    dcp_lh = pd_arr_lh[:, 6:8]
    pairs_lh_pa = pd_arr_lh.shape[0] != 0 and pd_arr_lh is not None

    # if the death time is inf, set it to 1.0
    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0
    
    return pd_lh, bcp_lh, dcp_lh, pairs_lh_pa

def getCriticalPoints3D_cr(likelihood, topo_dim):

    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=2, location="birth") # Same but maxdim=2
    pd_arr_lh = pd[pd[:, 0] == topo_dim] # 0 or 1-dim topological features
    pd_lh = pd_arr_lh[:, 1:3] # birth time and death time
    # birth critical points (3D coordinates)
    bcp_lh = pd_arr_lh[:, 3:6]
    # death critical points (3D coordinates) 
    dcp_lh = pd_arr_lh[:, 6:9]
    pairs_lh_pa = pd_arr_lh.shape[0] != 0 and pd_arr_lh is not None

    # if the death time is inf, set it to 1.0
    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0
    
    return pd_lh, bcp_lh, dcp_lh, pairs_lh_pa

def get_matchings(lh, gt):
    
    _, matchings = wasserstein_distance(lh, gt, matching=True)

    dgm_to_diagonal = matchings[matchings[:,1] == -1, 0]
    off_diagonal_match = np.delete(matchings, np.where(matchings == -1)[0], axis=0)

    return dgm_to_diagonal, off_diagonal_match


def getTopoLoss2d(pred_tensor, gt_tensor, topo_dim):
    if pred_tensor.ndim != 2:
        print("incorrct dimension")

    likelihood = pred_tensor.clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    if(np.min(likelihood) == 1 or np.max(likelihood) == 0): return 0.
    if(np.min(gt) == 1 or np.max(gt) == 0): return 0.
    
    # Get the critical points of predictions and ground truth
    pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints_cr(likelihood, topo_dim)
    pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints_cr(gt, topo_dim)

    # If the pairs not exist, continue for the next loop
    if not(pairs_lh_pa): return 0.
    if not(pairs_lh_gt): return 0.

    idx_holes_to_remove_for_matching, off_diagonal_for_matching = compute_dgm_force(pd_lh, pd_gt)

    idx_holes_to_remove = []
    off_diagonal_match = []

    if (len(idx_holes_to_remove_for_matching) > 0):
        for i in idx_holes_to_remove_for_matching:
            index_pd_lh_removed = np.where(np.all(pd_lh == pd_lh[i], axis=1))[0][0]
            idx_holes_to_remove.append(index_pd_lh_removed)
    
    if len(off_diagonal_for_matching) > 0:
        for idx, (i, j) in enumerate(off_diagonal_for_matching):
            index_pd_lh = np.where(np.all(pd_lh == pd_lh[i], axis=1))[0][0]
            index_pd_gt = np.where(np.all(pd_gt == pd_gt[j], axis=1))[0][0]
            off_diagonal_match.append((index_pd_lh, index_pd_gt))

    if (len(off_diagonal_match) > 0 or len(idx_holes_to_remove) > 0):
        for (idx, (hole_indx, j)) in enumerate(off_diagonal_match):
            if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                    bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                topo_cp_weight_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = 1 # push birth to the corresponding teacher birth i.e. min birth prob or likelihood
                topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = pd_gt[j][0]
            
            if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                    likelihood.shape[1]):
                topo_cp_weight_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = 1  # push death to the corresponding teacher death i.e. max death prob or likelihood
                topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = pd_gt[j][1]
        
        for hole_indx in idx_holes_to_remove:
            if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                    likelihood.shape[1]):
                topo_cp_weight_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = 1  # push to diagonal
                
                if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                    0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                        likelihood.shape[1]):
                    topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = \
                        likelihood[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                else:
                    topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = 1
            
            if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                    likelihood.shape[1]):
                topo_cp_weight_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = 1  # push to diagonal
                
                if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                    0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                        likelihood.shape[1]):
                    topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = \
                        likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                else:
                    topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((pred_tensor  * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()

    if printstuff:
        print("\nTopoloss: {}".format(loss_topo.item()))
        # print("\ntopo_cp_weight_map: {} shape ; {} min ; {} max\nunique: {}".format(topo_cp_weight_map.shape, np.min(topo_cp_weight_map), np.max(topo_cp_weight_map), np.unique(topo_cp_weight_map, return_counts=True)))
        # print("\ntopo_cp_ref_map AKA Image Intensity: {} shape ; {} min ; {} max\nunique: {}".format(topo_cp_ref_map.shape, np.min(topo_cp_ref_map), np.max(topo_cp_ref_map), np.unique(topo_cp_ref_map, return_counts=True)))

    return loss_topo

def getTopoLoss3d(pred_tensor, gt_tensor, topo_dim):
    """
    What I did was the same format as getTopoLoss2d but I added the extra dimesnions when 
    """
    if pred_tensor.ndim != 3:
        print("incorrect dimension - expected 3D tensor")
        return 0.

    likelihood = pred_tensor.clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    # Initialize weight and reference maps for 3D
    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    if(np.min(likelihood) == 1 or np.max(likelihood) == 0): return 0.
    if(np.min(gt) == 1 or np.max(gt) == 0): return 0.
    
    # Get the critical points of predictions and ground truth
    pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints3D_cr(likelihood, topo_dim)
    pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints3D_cr(gt, topo_dim)

    # If the pairs not exist, continue for the next loop
    if not(pairs_lh_pa): return 0.
    if not(pairs_lh_gt): return 0.

    idx_holes_to_remove_for_matching, off_diagonal_for_matching = compute_dgm_force(pd_lh, pd_gt)

    idx_holes_to_remove = []
    off_diagonal_match = []

    if (len(idx_holes_to_remove_for_matching) > 0):
        for i in idx_holes_to_remove_for_matching:
            index_pd_lh_removed = np.where(np.all(pd_lh == pd_lh[i], axis=1))[0][0]
            idx_holes_to_remove.append(index_pd_lh_removed)
    
    if len(off_diagonal_for_matching) > 0:
        for idx, (i, j) in enumerate(off_diagonal_for_matching):
            index_pd_lh = np.where(np.all(pd_lh == pd_lh[i], axis=1))[0][0]
            index_pd_gt = np.where(np.all(pd_gt == pd_gt[j], axis=1))[0][0]
            off_diagonal_match.append((index_pd_lh, index_pd_gt))

    if (len(off_diagonal_match) > 0 or len(idx_holes_to_remove) > 0):
        for (idx, (hole_indx, j)) in enumerate(off_diagonal_match):
            if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                    bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1] and int(
                    bcp_lh[hole_indx][2]) >= 0 and int(bcp_lh[hole_indx][2]) < likelihood.shape[2]):
                topo_cp_weight_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1]), int(bcp_lh[hole_indx][2])] = 1 # push birth to the corresponding teacher birth i.e. min birth prob or likelihood
                topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1]), int(bcp_lh[hole_indx][2])] = pd_gt[j][0]
            
            if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                    likelihood.shape[1] and int(dcp_lh[hole_indx][2]) >= 0 and int(dcp_lh[hole_indx][2]) <
                    likelihood.shape[2]):
                topo_cp_weight_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1]), int(dcp_lh[hole_indx][2])] = 1  # push death to the corresponding teacher death i.e. max death prob or likelihood
                topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1]), int(dcp_lh[hole_indx][2])] = pd_gt[j][1]
        
        for hole_indx in idx_holes_to_remove:
            if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                    likelihood.shape[1] and int(bcp_lh[hole_indx][2]) >= 0 and int(bcp_lh[hole_indx][2]) <
                    likelihood.shape[2]):
                topo_cp_weight_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1]), int(bcp_lh[hole_indx][2])] = 1  # push to diagonal
                
                if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                    0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                        likelihood.shape[1] and int(dcp_lh[hole_indx][2]) >= 0 and int(dcp_lh[hole_indx][2]) <
                        likelihood.shape[2]):
                    topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1]), int(bcp_lh[hole_indx][2])] = \
                        likelihood[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1]), int(dcp_lh[hole_indx][2])]
                else:
                    topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1]), int(bcp_lh[hole_indx][2])] = 1
            
            if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                    likelihood.shape[1] and int(dcp_lh[hole_indx][2]) >= 0 and int(dcp_lh[hole_indx][2]) <
                    likelihood.shape[2]):
                topo_cp_weight_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1]), int(dcp_lh[hole_indx][2])] = 1  # push to diagonal
                
                if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                    0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                        likelihood.shape[1] and int(bcp_lh[hole_indx][2]) >= 0 and int(bcp_lh[hole_indx][2]) <
                        likelihood.shape[2]):
                    topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1]), int(dcp_lh[hole_indx][2])] = \
                        likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1]), int(bcp_lh[hole_indx][2])]
                else:
                    topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1]), int(dcp_lh[hole_indx][2])] = 0

    # Convert back to tensors for GPU computation
    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((pred_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()

    if printstuff:
        print("\n3D Topoloss: {}".format(loss_topo.item()))

    return loss_topo