import torch
from torch import nn

# 创建的tensor dropout mask 和 最终 dropout 返回的结果       bsh + 2bsh 大小
def dropout_layer(x,dropout):
    assert 0 <= dropout <= 1 # 检查条件，如果dropout 不在（0，1）之间就终止条件
    if dropout == 0:
        return x 
    if dropout == 1:
        return torch.zeros_like(x)
    mask = (torch.Tensor(x.shape).uniform_(0,1)> dropout).float()
    return mask * x /(1.0-dropout) # 以概率 dropout置零数据，其余变量放大以保证期望不变


m = torch.arange(16,dtype=torch.float).reshape((2,8))
print(m)
a = dropout_layer(m,0.0)
b = dropout_layer(m,1.0)
c = dropout_layer(m,0.8)

d = 3
