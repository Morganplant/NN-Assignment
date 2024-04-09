import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from rich.logging import RichHandler
from rich.progress import Progress, track
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

FORMAT = "%(message)s"
log = logging.basicConfig(
    level="DEBUG", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")


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


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if os.path.exists("./data"):
        trainloader, valloader, testloader, classes = torch.load(DATA_DIR)
        return trainloader, valloader, testloader, classes
    os.mkdir("./data")
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    batch_size = 4

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, valloader, testloader, classes


def train_model(trainloader, valloader, testloader):
    net = Net()
    logging.info("Starting Training")
    epoch_limit = 50

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loss_values = []
    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    patience = 5
    wait = 0

    with Progress() as progress:
        training_progress = len(trainloader) * epoch_limit
        validation_progress = len(valloader) * epoch_limit
        analysis_progress = len(testloader) * epoch_limit

        training = progress.add_task("[red]Training...", total=training_progress)
        validation = progress.add_task(
            "[green]Validation...", total=validation_progress
        )
        analysis = progress.add_task("[blue]Analysis...", total=analysis_progress)

        for epoch in range(epoch_limit):
            net.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                progress.update(training, advance=1)
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    loss_values.append(round((running_loss / 2000), 3))
                    running_loss = 0.0

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in valloader:
                    progress.update(validation, advance=1)
                    inputs, labels = data
                    outputs = net(inputs)
                    val_loss += criterion(outputs, labels).item()
            val_loss /= len(valloader)

            if val_loss > prev_val_loss and save_count > 0:
                wait += 1
                if wait >= patience:
                    logging.debug(
                        f"Saving Current Best Model: {prev_val_loss} - {prev_val_loss}"
                    )
                    torch.save(net.state_dict(), f"{MODEL_DIR}/best_model.pt")
                    save_count = +1
            else:
                wait = 0

            prev_val_loss = val_loss

        logging.info("Finished Training")

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                progress.update(analysis, advance=1)
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.debug(
            f"Accuracy of the network on the 10000 test images: {100 * correct // total} % - [{correct}/{total}]"
        )

    return loss_values


def plot_training_loss(loss_values):
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Scatter(x=list(range(len(loss_values))), y=loss_values))
    fig.update_layout(
        title="Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    fig.show()


def main():
    trainloader, valloader, testloader, classes = load_data()
    loss_values = train_model(trainloader, valloader, testloader)
    plot_training_loss(loss_values)


if __name__ == "__main__":
    main()
