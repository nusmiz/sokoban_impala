import torch.nn as nn
import torch.nn.functional as F


class A3CModel(nn.Module):
    def __init__(self):
        super(A3CModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, 8, 4)
        self.conv_2 = nn.Conv2d(32, 64, 4, 2)
        self.conv_3 = nn.Conv2d(64, 64, 3, 1)
        self.l_1 = nn.Linear(2304, 512)
        self.l_pi = nn.Linear(512, 4)
        self.l_v = nn.Linear(512, 1)

    def forward(self, x):
        h = F.leaky_relu(self.conv_1(x))
        h = F.leaky_relu(self.conv_2(h))
        h = F.leaky_relu(self.conv_3(h))
        h = h.view(-1, 2304)
        h = F.leaky_relu(self.l_1(h))
        pi = self.l_pi(h)
        v = self.l_v(h)
        return pi, v

    def pi(self, x):
        h = F.leaky_relu(self.conv_1(x))
        h = F.leaky_relu(self.conv_2(h))
        h = F.leaky_relu(self.conv_3(h))
        h = h.view(-1, 2304)
        h = F.leaky_relu(self.l_1(h))
        pi = self.l_pi(h)
        return pi

    def v(self, x):
        h = F.leaky_relu(self.conv_1(x))
        h = F.leaky_relu(self.conv_2(h))
        h = F.leaky_relu(self.conv_3(h))
        h = h.view(-1, 2304)
        h = F.leaky_relu(self.l_1(h))
        v = self.l_v(h)
        return v
