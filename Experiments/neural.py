import torch
import torch.nn as nn
import torchvision.models as models


class BaselineNN(nn.Module):
    def __init__(self):
        super(BaselineNN, self).__init__()
        # Defining the network architecture
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(1024, 128)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Passing input through each layer of the network
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = x.view(-1, 64 * 4 * 4)  # Corrected flattening operation

        x = self.relu4(self.fc1(x))

        x = self.fc2(x)

        return x
