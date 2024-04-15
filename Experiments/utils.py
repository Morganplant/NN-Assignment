import logging
import os
from concurrent.futures import ProcessPoolExecutor
from time import sleep

import torch
import torchvision
import torchvision.transforms as transforms
from joblib import Parallel, delayed
from rich.logging import RichHandler
from rich.progress import Progress
from sklearn.model_selection import train_test_split

# Configure logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
stream_handler = RichHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

# Configure Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
GRAPH_DIR = os.path.join(ROOT_DIR, "graph")


def get_data_loaders(seed=42):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # Load CIFAR-10 dataset
    do_download = not os.path.exists(DATA_DIR)
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=do_download, transform=transform
    )

    # Split training data into training and validation sets with the specified seed
    train_data, val_data = train_test_split(
        train_dataset, test_size=0.2, random_state=seed
    )

    # Create data loaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False)
    return train_loader, val_loader


def train_batch(model, optimizer, criterion, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()


def train(progress, model, optimizer, criterion, data_loader, epoch_limit):
    train_progress = progress.add_task(
        "[red]Training...", total=(epoch_limit * len(data_loader))
    )

    for _ in range(epoch_limit):
        model.train()
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(train_batch, model, optimizer, criterion, data, target)
                for data, target in data_loader
            ]
            for future in futures:
                future.result()
                progress.update(train_progress, advance=1)

    progress.update(train_progress, visible=False)


def evaluate(progress, model, data_loader):
    model.eval()
    correct = 0
    total = 0
    validate_progress = progress.add_task(
        "[green]Validating...", total=len(data_loader)
    )

    with torch.no_grad():
        for data, target in data_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            progress.update(validate_progress, advance=1)

    progress.update(validate_progress, visible=False)
    return correct / total
