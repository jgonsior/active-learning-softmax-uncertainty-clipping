import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LeNet(nn.Module):
    def __init__(self,number_classes,dropout=False):
        super().__init__()
        self.use_dropout = dropout
        #self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        #self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        #self.fc0 = nn.Linear(768, 768)
        self.fc1 = nn.Linear(768, number_classes)

    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 1))
        #x = F.relu(F.max_pool2d(self.conv2(x), 1))
        #x = x.view(x.size()[0], -1)
        #x = F.relu(self.fc0(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x
