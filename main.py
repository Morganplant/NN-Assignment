import logging
import os

import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from plotly.subplots import make_subplots
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
GRAPH_DIR = os.path.join(ROOT_DIR, "graph")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(
            64 * 8 * 8, 128
        )  # 8x8 because of two max poolings (32x32 -> 16x16 -> 8x8)
        self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the input for fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        # Forward pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiate the model
model = Net()
# Print the model architecture
# logging.debug(model)


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if os.path.exists("./data"):
        logging.warning(f"directory exists - '{DATA_DIR}'")
        exit()
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
    logging.debug(f"Trainloader : {len(trainloader)}")
    logging.debug(f"Vallloader : {len(valloader)}")
    logging.debug(f"Testloader : {len(testloader)}")

    return trainloader, valloader, testloader, classes


def train_model(trainloader, valloader, testloader):
    # Setup
    epoch_limit = 50
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Setup Progress Bars

    def training(epoch_limit, trainloader):
        loss_values = []

        logging.info("Starting Training")
        with Progress() as progress:
            training_bar = progress.add_task(
                "[red]Training...", total=(len(trainloader) * epoch_limit)
            )
            for _ in range(epoch_limit):
                net.train()
                for _i, data in enumerate(trainloader, 0):
                    progress.update(training_bar, advance=1)
                    running_loss = 0.0
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    loss_values.append(running_loss)
                    # if i % 2000 == 1999:  # Log the loss every 2000 mini-batches
                    #     logging.debug(f"Training loss: {running_loss}")
            return loss_values

    def validation(epoch_limit, valloader):
        val_loss_values = []
        prev_val_loss = float("inf")
        saved_model_path = os.path.join(MODEL_DIR, "best_model.pt")

        net.eval()
        with Progress() as progress:
            validation_bar = progress.add_task(
                "[green]Validation...", total=(len(valloader) * epoch_limit)
            )
            for _ in range(epoch_limit):
                with torch.no_grad():
                    for data in valloader:
                        progress.update(validation_bar, advance=1)
                        val_loss = 0.0  # Reset val_loss for each validation iteration
                        inputs, labels = data
                        outputs = net(inputs)
                        val_loss += criterion(outputs, labels).item()
                        val_loss_values.append(val_loss)
                        # if i % 2000 == 1999:  # Log the loss every 2000 mini-batches
                        #     logging.debug(f"Validation loss: {val_loss}")

                    val_loss /= len(valloader)  # Calculate the average loss

                    # Save Current Best if there overfitting starts:
                    if val_loss > prev_val_loss and not os.path.isfile(
                        saved_model_path
                    ):
                        logging.debug(f"Saving Current Best Model: {val_loss}")
                        torch.save(net.state_dict(), saved_model_path)
                    prev_val_loss = val_loss
        return val_loss_values

    def accuracy_calc(testloader):
        accuracy_values = []

        correct = 0
        total = 0
        with Progress() as progress:
            analysis_bar = progress.add_task(
                "[blue]Analysis...", total=(len(testloader) * epoch_limit)
            )
            with torch.no_grad():
                for data in testloader:
                    progress.update(analysis_bar, advance=1)
                    images, labels = data
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    accuracy = 100 * correct / total
                    accuracy_values.append(accuracy)
        return accuracy_values

    training_values = training(epoch_limit, trainloader)
    validation_values = validation(epoch_limit, valloader)

    # trim training_values down to the same length as validation_values
    trimmed_training_values = training_values[: len(validation_values)]

    accuracy_values = accuracy_calc(testloader)
    logging.debug(
        f"training: {len(training_values)}\nvalidation: {len(validation_values)}\naccuracy: {len(accuracy_values)}"
    )

    df = pd.DataFrame(
        {
            "Trimmed Training Loss": trimmed_training_values,
            "Validation Loss": validation_values,
            "Accuracy": accuracy_values,
            "Epoch": range(len(trimmed_training_values)),
        }
    )
    return df, training_values


def plot_data(df, training_values):
    logging.info("plotting Data")
    logging.debug("Adding trimmed Training Loss")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Epoch"], y=df["Trimmed Training Loss"], name="Training Loss"),
    )
    logging.debug("Adding Validation Loss")
    fig.add_trace(
        go.Scatter(x=df["Epoch"], y=df["Validation Loss"], name="Validation Loss"),
    )
    fig.write_html(os.path.join(GRAPH_DIR, "Loss.html"))
    logging.debug("Adding Accuracy")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Epoch"], y=df["Accuracy"], name="Accuracy"),
    )
    fig.write_html(os.path.join(GRAPH_DIR, "Accuracy.html"))
    fig = go.Figure()

    logging.debug("Adding Full Training Data")
    fig.add_trace(
        go.Scatter(
            x=list(range(len(training_values))),
            y=training_values,
            name="Full Training Data",
        )
    )
    fig.write_html(os.path.join(GRAPH_DIR, "full_training_data.html"))


def main():
    trainloader, valloader, testloader, classes = load_data()
    df, training_values = train_model(trainloader, valloader, testloader)
    plot_data(df, training_values)


if __name__ == "__main__":
    main()
