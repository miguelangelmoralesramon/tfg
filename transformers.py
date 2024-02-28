#codigo de la arquitectura de un transformer, basado en el paper:
#Attention is all you need

import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q,k,v,mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1,0,2,3) + mask
        scaled = scaled.permute(1,0,2,3)
    attention = F.softmax(scaled,dim=-1)
    values = torch.matmul(attention,v)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
    
    def forward(self):
        even_i = torch.arange(0,self.d_model,2).float()
        denom = torch.pow(10000,even_i/self.d_model)
        pos = (torch.arange(self.seq_length).reshape(self.seq_length,1))
        even_PE = torch.sin(pos / denom)
        odd_PE = torch.cos(pos / denom)
        stacked = torch.stack([even_PE,odd_PE],dim=2)
        PE = torch.flatten(stacked,start_dim=1,end_dim=2)
        return PE
    
class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self,seq_length,d_model,lang_to_ind,START_TOKEN,END_TOKEN,PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(lang_to_ind)
        self.seq_length = seq_length
        self.embedding = nn.Embedding(self.vocab_size,d_model)
        self.lang_to_ind = lang_to_ind
        self.pe = PositionalEncoding(d_model,seq_length)
        self.dropout = nn.Dropout(p=0.2)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_normalize(self,batch,start_token,end_token):
        def tokenize(sentence,start_token,end_token):
            word_ind = [self.lang_to_ind[token] for token in list(sentence)]
            if start_token:
                word_ind.insert(0,self.lang_to_ind[self.START_TOKEN])
            if end_token:
                word_ind.append(self.lang_to_ind[self.END_TOKEN])
            for _ in range(len(word_ind), self.seq_length):
                word_ind.append(self.lang_to_ind[self.PADDING_TOKEN])
            return torch.tensor(word_ind)
        
        tokenized = []
        for sentence in range(len(batch)):
            tokenized.append(tokenize(batch[sentence],start_token,end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self,x,start_token,end_token):
        x = self.batch_tokenize(x,start_token,end_token)
        x = self.embedding(x)
        pos = self.pe().to(get_device())
        x = self.dropout(x+pos)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model%num_heads==0, "The model dim must be divisible by the number of heads"
        self.qkv_layer = nn.Linear(d_model,3*d_model)
        self.linear_layer = nn.Linear(d_model,d_model)

    def forward(self,x,mask):
        batch_size,seq_length,d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size,seq_length,self.num_heads,3*self.head_dim)
        qkv = qkv.permute(0,2,1,3)
        q,k,v = qkv.chunk(3,dim=-1)
        values,attention = scaled_dot_product(q,k,v,mask)
        values = values.permute(0,2,1,3).reshape(batch_size,seq_length,self.num_heads*self.head_dim)
        out = self.linear_layer
        return out
    
class LayerNormalization(nn.Module):
    def __init__(self,param_shape,eps=1e-6):
        super().__init__()
        self.param_shape = param_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(param_shape))
        self.beta = nn.Parameter(torch.zeros(param_shape))

    def forward(self,inputs):
        dims = [-(i+1) for i in range(len(self.param_shape))]
        mean = inputs.mean(dim=dims,keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out
    
class PositionFeedForward(nn.Module):
    def __init__(self,d_model,hidden,drop_prob=0.1):
        super(PositionFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,hidden)
        self.linear2 = nn.Linear(hidden,d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
