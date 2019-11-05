import torch
import torch.nn as nn
from torchvision import models
from torch.nn.modules import Module

class CNNNet(nn.Module):
    def __init__(self, temp, code_length, n_class, pretrained=True):
        super(CNNNet, self).__init__()
        self.temp = temp
        original_model = models.alexnet(pretrained)
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = nn.Sequential(
            *list(original_model.classifier.children())[:-1]
        )
        self.extra = nn.Linear(4096, code_length)
        self.extra2 = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(code_length, n_class)
        )

    def bottleneck(self, h):
        eps = torch.rand_like(h)
        z = (torch.clamp(eps, min=1e-20).log() - torch.clamp((1-eps),min=1e-20).log() + h) / self.temp
        return z

    def forward(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        f = f.view(f.size(0), 256 * 6 * 6)
        f = self.classifier(f)
        f = self.extra(f)       #log m
        z = self.bottleneck(f)  #z
        y = self.extra2(z)      #y
        return y, z, f

class CNNExtractNet(nn.Module):
    def __init__(self, original_model):
        super(CNNExtractNet, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = original_model.classifier
        self.extra = original_model.extra

    def forward(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        f = f.view(f.size(0), 256 * 6 * 6)
        f = self.classifier(f)
        y = self.extra(f)
        return y
