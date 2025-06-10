import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


print(np.array([1,2,3,4]).dtype)
print(np.array([1,2,3,4]).astype(np.float32).dtype)
nparr = np.array([1,2,3,4]).astype(np.float32).reshape(2,2)
print(f"nparr = {nparr}")
print(np.block([[nparr,nparr],[nparr,nparr]]))
nparr = np.block([[nparr,nparr],[nparr,nparr]]).reshape(1,1,4,4)
print(f"nparr = {nparr}")
input = torch.from_numpy(nparr).clone()
print(f"input = {input}")
