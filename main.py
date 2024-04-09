import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using {str(device).upper()}")

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 training set
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split the training set into training and validation sets
# Assuming 80% for training and 20% for validation
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

# Define batch size
batch_size = 4

# Create data loaders for training, validation, and test sets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Assuming you have a validation loader named 'valloader'
# Assuming 'net', 'criterion', and 'optimizer' are defined elsewhere

loss_values = []

best_val_loss = float('inf')
prev_val_loss = float('inf')
patience = 5  # Number of epochs to wait before reducing learning rate
wait = 0

epoch_limit = 50

# Set the number of workers for data loading
num_workers = 4  # Adjust this number based on your CPU's capacity

for epoch in tqdm(range(50)):  # Number of maximum epochs
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):  # enumerate over trainloader
        inputs, labels = data  # unpack data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            loss_values.append(round((running_loss / 2000), 3))
            running_loss = 0.0

    # Validation loss calculation
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            outputs = net(inputs)
            val_loss += criterion(outputs, labels).item()
    val_loss /= len(valloader)

    # Check for early stopping condition
    if val_loss > prev_val_loss:
        wait += 1
        if wait >= patience:
            print(f'Early stopping at epoch {epoch + 1} due to validation loss increase or plateauing.')
            break
    else:
        wait = 0

    prev_val_loss = val_loss

    # Save the model if validation loss improved
    if val_loss < best_val_loss:
        torch.save(net.state_dict(), 'model/best_model.pt')
        best_val_loss = val_loss

print('Finished Training')


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} % - [{correct}/{total}]')

import plotly.graph_objects as go
fig = go.Figure(
    data=go.Scatter(x=list(range(len(loss_values))), y=loss_values)
)
fig.update_layout(
    title="Training Loss",
    xaxis_title="Epoch",
    yaxis_title="Loss",
)
