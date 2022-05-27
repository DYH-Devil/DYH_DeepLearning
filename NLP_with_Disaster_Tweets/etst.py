import torch.nn as nn
import torch
import torch.nn.functional as F

query = nn.Parameter(torch.Tensor(5 , 5))
nn.init.uniform_(query , -0.1 , 0.1)
output = torch.randn([64 , 100 , 5])#[batchsize , seqlen , hidden]
query_trix = torch.matmul(output , query)
print("querytrix : " , query_trix.shape)#[64 , 100 , 5]====[batch , seqlen , hidden]得到key矩阵


key = nn.Parameter(torch.Tensor(5))
nn.init.uniform_(key , -0.1 , 0.1)
key_trix = torch.matmul(query_trix , key)
print("key_trix : " , key_trix.shape)


alpha = F.softmax(key_trix , dim = -1).unsqueeze(-1)
print("alpha : " , alpha.shape)#[batch , seqlen , 1]query对应的每个key的a值，即相关性

out = alpha * output
print("out shape : " , out.shape)#注意力矩阵[batch , seqlen , hidden]
out = torch.sum(out, dim=1)
print(out.shape)