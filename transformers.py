#codigo de la arquitectura de un transformer, basado en el paper:
#Attention is all you need

import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def producto_escalar(q,k,v,mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1,0,2,3) + mask
        scaled = scaled.permute(1,0,2,3)
    attention = F.softmax(scaled,dim=-1)
    values = torch.matmul(attention,v)
    return values, attention