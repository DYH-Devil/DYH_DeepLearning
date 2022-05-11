import numpy as np
import torch
import torch.nn.functional as F

from imdbDataset_1 import dataLoader
from lib import device , test_batch_size
from model_4 import MyModel
from tqdm import tqdm
import os
from torch.optim import Adam

imdb_model = MyModel()
imdb_model = imdb_model.to(device)

optimizer = Adam(imdb_model.parameters() , lr = 0.001)#优化器类实例化

if os.path.exists('./model/model.pkl') :
    imdb_model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))
    #加载模型与优化器类

def eval():
    loss_list = []
    acc_list = []
    test_loader = dataLoader(train = False , batch_size = test_batch_size)
    for idx,(input,target) in enumerate(test_loader):
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = imdb_model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append((cur_loss.cpu().item()))
            # 计算准确率
            pred = output.max(dim=-1)[1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())

    print("total loss,acc:",np.mean(loss_list),np.mean(acc_list))

if __name__ == '__main__':
    eval()
