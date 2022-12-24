import torch 
import torch.nn as nn

import torch.optim
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from tqdm.notebook import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

from dataloader import train_data,valid_data,test_data
from loss import dice_coef_metric, compute_iou, Dcs
from unet import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(str(device)+'is available')

def trans(img,shape):
      x = img.shape[2]
  
      u = (x-shape)/2
     
      a = int(u-1)
      b = int(a+shape)
      return img[:,:,a:b,a:b]


def training(model,criterion,optimizer,num_ep,train_loader,val_loader):
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_ep):
        model.train()
        print('epoch:',epoch)
        losses = []
        train_iou = []
        for inputs, labels in train_loader:
            # print('loader: ',i_step)
            inputs = inputs.to(device)
            # print(inputs.shape)
            labels = labels.to(device)
            outputs = model(inputs)
            s = outputs.shape[2]
            labels = trans(labels,s)
            print('lab',labels.shape)
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            print('out',outputs.shape)
            optimizer.zero_grad()
            loss = criterion.forward(labels,outputs)
            losses.append(loss.item())
            train_dcs = dice_coef_metric(out_cut, labels.data.cpu().numpy())
            train_iou.append(train_dcs)
            loss.backward()
            optimizer.step()
            
        val_mean_iou = compute_iou(model, val_loader)
        
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)

        print("Epoch [%d]" % (epoch))
        print("Mean loss on train:", np.array(losses).mean(), 
              "\nMean DICE on train:", np.array(train_iou).mean(), 
              "\nMean DICE on validation:", val_mean_iou)
        
    return loss_history, train_history, val_history
            

u_n = UNet(init_features=64).to(device)
train_df, train_loader = train_data()
valid_df,valid_loader = valid_data()
test_df,test_loader = test_data()


optimizer_ft = torch.optim.Adamax(u_n.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


num_ep = 50
