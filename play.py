import torch
import torch.nn.functional as F


t=torch.tensor([[1,2],[3,4],[5,6]])
print(t)
r=t[torch.randperm(t.size()[0])]
r=t[torch.randperm(t.size()[0])]
print(r)