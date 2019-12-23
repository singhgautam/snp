import torch
from torch import nn
from torch.nn import functional as F

'''
    Copied and modified from https://github.com/masa-su/pixyzoo/tree/gqn/GQN
'''

class Pyramid(nn.Module):
    def __init__(self, nc, nc_query):
        super(Pyramid, self).__init__()
        self.nc = nc
        self.nc_query = nc_query
        self.net = nn.Sequential(
            nn.Conv2d(nc+nc_query, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=8, stride=8),
            nn.ReLU()
        )
        

    def forward(self, x, v):
        # Broadcast
        v = v.view(-1, self.nc_query, 1, 1).repeat(1, 1, 64, 64)
        r = self.net(torch.cat((v, x), dim=1))
        
        return r

    def get_output_size(self):
        return 256, 1, 1
    
class Tower(nn.Module):
    def __init__(self, nc, nc_query):
        super(Tower, self).__init__()
        self.nc = nc
        self.nc_query = nc_query

        self.conv1 = nn.Conv2d(nc, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256+nc_query, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256+nc_query, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

    def get_output_size(self):
        return 256, 16, 16

    def forward(self, x, v):
        # Resisual connection
        skip_in  = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        # Broadcast
        v = v.view(v.size(0), self.nc_query, 1, 1).repeat(1, 1, 16, 16)
        
        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))

        return r

class Pool(nn.Module):
    def __init__(self, nc, nc_query):
        super(Pool, self).__init__()
        self.nc = nc
        self.nc_query = nc_query
        self.conv1 = nn.Conv2d(nc, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256+nc_query, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256+nc_query, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.pool  = nn.AvgPool2d(16)

    def forward(self, x, v):
        # Resisual connection
        skip_in  = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        # Broadcast
        v = v.view(v.size(0), self.nc_query, 1, 1).repeat(1, 1, 16, 16)
        
        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))
        
        # Pool
        r = self.pool(r)

        return r

    def get_output_size(self):
        return 256, 1, 1