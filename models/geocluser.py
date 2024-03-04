import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.nn import functional as f



# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self,opt, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False, temp=torch.randn(1, 64, 30, 30)):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.opt = opt
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.mask_ = self.generate_mask(temp)
        self.mask = self.mask_.unsqueeze(0).unsqueeze(0).expand(temp.shape[0], self.num_clusters, temp.shape[2], temp.shape[3])

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def hard_assign(self, soft_assign):
        N, num_clusters, features = soft_assign.shape[0:3]
        first_index = torch.zeros((N * features, 3), dtype=torch.int, device=soft_assign.device)
        second_index = torch.zeros((N * features, 3), dtype=torch.int, device=soft_assign.device)
        indices = torch.arange(N * features)
        first_assign = torch.argmax(soft_assign, 1)
        second_assign = torch.argsort(soft_assign, 1)[:, -2]

        first_index[:, 2] = first_assign.reshape(-1)
        second_index[:, 2] = second_assign.reshape(-1)
        first_index[:, 0] = indices // features
        first_index[:, 1] = indices % features
        second_index[:, 0] = indices // features
        second_index[:, 1] = indices % features
        return first_index.to(torch.long), second_index.to(torch.long)

        # return first_index.to(torch.long)


    def generate_mask(self, x):
        mask = torch.zeros([x.shape[2], x.shape[3]], dtype=x.dtype, layout=x.layout, device=x.device)
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                mask[i][j] = min(min(x.shape[2] - 1 - i, i), min(x.shape[3] - 1 - j, j))
        mask = torch.mul(mask, mask)
        mask = torch.mul(mask, mask)
        return mask

    def forward(self, x):
        x = x.unsqueeze(0)
        N, C, H, W = x.shape[:4]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # first_index = self.hard_assign(soft_assign)
        first_index, second_index = self.hard_assign(soft_assign)
        soft_assign_HW = soft_assign.view(N, self.num_clusters, H, W)

        windows_all = f.unfold(soft_assign_HW, kernel_size=(3, 3), stride=(1, 1), padding=1)
        B, C_kh_kw, L = windows_all.size()
        windows_all = windows_all.permute(0, 2, 1).view(N, L, self.num_clusters, 3*3)
        weight_scr = torch.zeros_like(windows_all)
        #
        first_values = windows_all[first_index[:, 0], first_index[:, 1], first_index[:, 2]]
        weight_scr[first_index[:, 0], first_index[:, 1], first_index[:, 2]] = first_values

        second_values = windows_all[second_index[:, 0], second_index[:, 1], second_index[:, 2]]
        weight_scr[second_index[:, 0], second_index[:, 1], second_index[:, 2]] = second_values

        weight_scr = weight_scr.view(N, L, C_kh_kw).permute(0, 2, 1)
        weight_scr = F.fold(input=weight_scr, output_size=(H, W),
                               kernel_size=(3, 3), stride=(1, 1), padding=1)

        # weight_scr = torch.mul(weight_scr, self.mask)
        weight_scr = torch.mul(weight_scr, self.mask)
        soft_assign = weight_scr.view(N, self.num_clusters, -1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)       # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        # print(vlad)
        return vlad