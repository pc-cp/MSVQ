import torch.nn as nn
import torch

class MoCoHead(nn.Module):
    def __init__(self, input_dim=2048, out_dim=128, hiddent_dim=2048):
        super().__init__()
        self.linear = nn.Linear(input_dim, hiddent_dim)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(hiddent_dim, out_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.out(x)
        return x

class LinearHead(nn.Module):
    def __init__(self, net, dim_in=2048, num_class=1000):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(dim_in, num_class)

        for param in self.net.parameters():
            param.requires_grad = False

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            feat = self.net(x)
        return self.fc(feat)


