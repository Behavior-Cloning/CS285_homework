import torch
from torch import nn
from torch.nn import functional as F

x=torch.arange(4)
y=torch.zeros(4)
torch.save([x,y],'x-file')

x2,y2=torch.load("x-file")
print((x2,y2))