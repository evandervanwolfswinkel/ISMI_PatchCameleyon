import torch
import torch.nn as nn

class NetFullConv(torch.nn.Module):
    def __init__(self):
        super(NetFullConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((4,4)),
            nn.ELU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((4,4)),
            nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Conv2d(64,32,4),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32,1,1)
        )

    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        x = self.fc(x)
        x = x.squeeze(3).squeeze(2)
        #print(x.shape)
        return torch.sigmoid(x)

class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5)
        self.bn6 = nn.BatchNorm2d(512)
        self.drop = nn.Dropout(0.2)
        self.drop2d = nn.Dropout2d(0.2)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(18432, 5012)
        self.bn1d1 = nn.BatchNorm1d(5012)
        self.lin2 = nn.Linear(5012, 1024)
        self.bn1d2 = nn.BatchNorm1d(1024)
        self.lin3 = nn.Linear(1024, 1)
        self.relu = nn.ELU()
        self.maxpool = nn.MaxPool2d((2,2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.drop2d(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.drop2d(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)
        x = self.flat(x)
        x = self.relu(self.bn1d1(self.lin1(x)))
        x = self.drop(x)
        x = self.relu(self.bn1d2(self.lin2(x)))
        x = self.lin3(x)
        y = self.sigmoid(x)
        return y

class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5)
        self.bn6 = nn.BatchNorm2d(512)
        self.drop = nn.Dropout(0.2)
        self.drop2d = nn.Dropout2d(0.2)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(18432, 5012)
        self.bn1d1 = nn.BatchNorm1d(5012)
        self.lin2 = nn.Linear(5012, 1024)
        self.bn1d2 = nn.BatchNorm1d(1024)
        self.lin3 = nn.Linear(1024, 1)
        self.relu = nn.ELU()
        self.maxpool = nn.MaxPool2d((2,2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.drop2d(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.drop2d(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)
        x = self.flat(x)
        x = self.relu(self.bn1d1(self.lin1(x)))
        x = self.drop(x)
        x = self.relu(self.bn1d2(self.lin2(x)))
        x = self.lin3(x)
        y = self.sigmoid(x)
        return y
        
class NetFullConvOld(torch.nn.Module):
    def __init__(self):
        super(NetFullConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5)
        self.bn6 = nn.BatchNorm2d(512)
        self.drop = nn.Dropout(0.2)
        self.drop2d = nn.Dropout2d(0.2)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(18432, 5012)
        self.bn1d1 = nn.BatchNorm1d(5012)
        self.lin2 = nn.Linear(5012, 1024)
        self.bn1d2 = nn.BatchNorm1d(1024)
        self.lin3 = nn.Linear(1024, 1)
        self.relu = nn.ELU()
        self.maxpool = nn.MaxPool2d((2,2))

        self.c = nn.Conv2d(512,512,6)
        self.bnc = nn.BatchNorm2d(512)
        self.d = nn.Conv1d(512, 128, 1)
        self.bnd = nn.BatchNorm1d(128)
        self.e = nn.Conv1d(128, 1, 1)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.drop2d(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.drop2d(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)
        x = self.relu(self.bnc(self.c(x)))
        x = self.relu(self.bnd(self.d(x.squeeze(3))))
        x = self.e(x)
        # x = self.flat(x)
        # x = self.relu(self.bn1d1(self.lin1(x)))
        # x = self.drop(x)
        # x = self.relu(self.bn1d2(self.lin2(x)))
        # x = self.lin3(x)
        # y = self.sigmoid(x)
        #print(x.shape)
        return torch.sigmoid(x.squeeze(2))
