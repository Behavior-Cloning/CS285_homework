import torch

A=torch.tensor([[1.0,0.3,0.02],
                [0.3,1,0],
                [0.02,0,1]])


L=torch.cholesky(A)
print(L)
print(torch.mm(L,L.t()))