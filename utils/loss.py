import torch.nn as nn
import torch
from torch.autograd import Variable
import pdb
import numpy as np

class proposed(nn.Module):
    def __init__(self, alpha, code_length, temp1, temp2):
        super(proposed, self).__init__()
        self.alpha = alpha
        self.code_length = code_length
        self.temp1 = Variable(torch.from_numpy(np.array(temp1)).type(torch.FloatTensor).cuda(),requires_grad=False)
        self.temp2 = Variable(torch.from_numpy(np.array(temp2)).type(torch.FloatTensor).cuda(),requires_grad=False)

    def forward(self, x, z, m, y):
        log_f = self.temp2.log() - self.temp2 * z - 2 *(1 + torch.clamp((-self.temp2 * z),max=88.7).exp()).log()
        log_g = self.temp1.log() - self.temp1 * z + m - 2 *(1 + torch.clamp((-self.temp1 * z + m),max=88.7).exp()).log()
        kld =  (log_g - log_f).sum()
        x_hat = torch.sigmoid(x)
        data_loss = (-y * torch.clamp(x_hat,min=1e-20).log() - (1-y) * torch.clamp(1-x_hat,min=1e-20).log()).sum()
        loss = data_loss + self.alpha*kld
        if torch.isnan(loss):
            pdb.set_trace()
            # use torch.clamp() if you still get over/underflow
        return loss
