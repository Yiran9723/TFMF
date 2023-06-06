import Constants as Constants
import torch
import torch.nn as nn
import numpy as np

# a = [[1,1],[1,0],[0,1]]
# # b = [[2,2],[2,2],[2,2]]
# # a = [1,0]
# b = [3,3]
#
# a = torch.from_numpy(np.ndarray(a)).float().cuda()
# b = torch.from_numpy(np.ndarray(b)).float().cuda()
# len_q = b.size(1)
# padding_mask = a.eq(Constants.PAD)
# padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
# print("padding_mask", padding_mask)
# print("Constants.PAD:",Constants.PAD)

a = torch.triu(  # 返回一个上三角矩阵
        torch.ones((5, 5), dtype=torch.uint8), diagonal=1)
print(a)
print(a.shape)
subsequent_mask = a.unsqueeze(0).expand(4, -1, -1)
print(subsequent_mask)
print(subsequent_mask.shape)
# b = torch.ones((8, 5), dtype=torch.uint8)
# sz_b, len_s = b.size()
# print("sz:",sz_b)
# print("len:",len_s)