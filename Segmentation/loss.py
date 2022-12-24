import torch
import torch.nn as nn
import numpy as np

def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target*inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0 
    return intersection/union

class Dcs(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dcs, self).__init__()
        self.smooth =1

    def forward(self, y_true,y_pred):


        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        # intersection = (inputs * targets).sum()                            
        # dice = (2.*intersection + 1)/(inputs.sum() + targets.sum() + 1)  

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + 1.0)/(inputs.sum() + targets.sum() + 1.0)  
        
        return 1 - dice

def compute_iou(model, loader, threshold=0.3):
    valloss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    with torch.no_grad():

        for i_step, (data, target) in enumerate(loader):
            
            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0
            picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += picloss

    return valloss / i_step

# a = torch.rand(26, 1, 266, 266)
# b = torch.rand(26,1,266,266)

# d = Dcs()
# print(d.forward(a,b))