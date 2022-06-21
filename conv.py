from scipy import signal
import numpy as np
import torch
from torch import nn, from_numpy, Tensor, long
import torch.nn.functional as f
from torch.nn import BCELoss
from torch import nn
from math import log


class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=0.4, gamma=2):
            super(FocalLoss, self).__init__()        
            self.alpha = torch.tensor([alpha, 1-alpha])   
            self.gamma = gamma
            
    def forward(self, inputs, targets):
            BCE_loss = f.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
            targets = targets.type(long)        
            at = self.alpha.gather(0, targets.data.view(-1))        
            pt = torch.exp(-BCE_loss)        
            F_loss = at*(1-pt)**self.gamma * BCE_loss        
            return F_loss.mean()

def conv():
    relu = nn.ReLU(inplace=True)

    data = np.array([[5,6,0,1,8,2],
                     [2,5,7,2,3,7],
                     [0,7,2,4,5,6],
                     [5,3,6,9,3,1],
                     [6,5,3,1,4,6],
                     [5,2,4,0,8,7]])
    nurcle1 = np.array([[ 1,-1, 0],
                        [-1, 1,-1],
                        [ 0,-1, 1]])
    nurcle2 = np.array([[-1, 2,-1],
                        [ 1, 5, 1],
                        [-1, 0,-1]])

    full = signal.convolve2d(data, nurcle1)
    same = signal.convolve2d(data, nurcle2, 'same')
    valid = signal.convolve2d(data, nurcle1, 'valid')

    # print(relu(from_numpy(full)))
    # print('-'*60)
    print(relu(from_numpy(same)))
    # print('-'*60)
    # print(relu(from_numpy(valid)))


def multi_cross():
    label = Tensor([4])
    criterion = nn.CrossEntropyLoss()
    src1 = f.softmax(Tensor([0.01,-0.01,-0.05,0.02,0.1]), 0)
    # src2 = f.softmax(Tensor([log(10), log(30), log(50), log(90)]), 0)
    src1 = from_numpy(np.array([src1.numpy()]))
    # src2 = from_numpy(np.array([src2.numpy()]))
    print(src1)
    # print(src2)
    loss1 = criterion(src1, label.long())
    # loss2 = criterion(src2, label.long())
    print(loss1)
    # print(loss2)

def bin_cross():
    label = Tensor([0, 1, 0, 0, 1])
    src = Tensor([0.2, 0.8, 0.4, 0.1, 0.9])
    bce = BCELoss()
    loss = bce(src, label)
    print(loss)
    focal = FocalLoss(alpha=0.4, gamma=2)
    loss = focal(src, label)
    print(loss)

def mae():
    from sklearn import metrics
    y = np.array([0.3,0.4,0.5,0.6,0.7,0.8])
    y_hat = np.array([0.42,0.46,0.53,0.58,0.7,0.88])
    MSE = metrics.mean_squared_error(y, y_hat)
    RMSE = metrics.mean_squared_error(y, y_hat)**0.5
    MAE = metrics.mean_absolute_error(y, y_hat)
    MAPE = metrics.mean_absolute_percentage_error(y, y_hat)
    print(MAE)
    print(MSE)


if __name__ == "__main__":
    # bin_cross()
    # conv()
    # multi_cross()
    mae()