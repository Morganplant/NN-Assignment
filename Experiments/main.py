# main.py
import random

import torch.nn as nn
import torch.optim as optim
from neural import BaselineNN
from rich.progress import (MofNCompleteColumn, Progress, SpinnerColumn,
                           TimeElapsedColumn)
from utils import evaluate, get_data_loaders, log, train


def main():
    criterion = nn.CrossEntropyLoss()
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    mean_performance = {}

    # Setup Progress Bars:
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:

        for lr in learning_rates:
            # Setup Progress Bar:
            lr_progress = progress.add_task(f"[cyan]Learning Rate: [{lr}]...", total=5)
            performance_lr = []
            for iterationn in range(5):  # Run the model at least five times
                # trunk-ignore(bandit/B311)
                seed = random.randint(1, 1000)
                random.seed(seed)  # Set the random seed
                log.debug(f"Using Seed: {seed} for {lr} at {iterationn}")

                # Create model, optimizer, and data loaders
                model = BaselineNN()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                train_loader, val_loader = get_data_loaders(
                    seed
                )  # Pass seed to data loaders

                # Train the model
                train(
                    progress, model, optimizer, criterion, train_loader, epoch_limit=3
                )

                # Evaluate the model
                performance = evaluate(
                    progress, model, val_loader
                )  # Implement evaluate function to calculate performance

                performance_lr.append(performance)
                progress.update(lr_progress, advance=1)

            mean_performance[lr] = {
                "Average": sum(performance_lr) / len(performance_lr),
                "Perfomance": performance_lr,
            }

    print("Mean Performance for each Learning Rate:")
    for lr, performance in mean_performance.items():
        print(f"Learning Rate: {lr}, Mean Performance: {performance}")


if __name__ == "__main__":
    main()

# Mean Performance for each Learning Rate:
# Learning Rate: 0.1, Mean Performance: 0.1001
# Learning Rate: 0.01, Mean Performance: 0.09518
# Learning Rate: 0.001, Mean Performance: 0.22949999999999998
# Learning Rate: 0.0001, Mean Performance: 0.3266
# Learning Rate: 1e-05, Mean Performance: 0.30536
