import torch
import torch.nn.functional as F

target = torch.ones([10, 64], dtype=torch.float32)
output = torch.full([10, 64], 0.9) 
pos_weight = torch.ones([64])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(output, target)
print(loss)