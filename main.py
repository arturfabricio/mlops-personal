import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

@click.group()
def cli():
    pass

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training MNIST dataset...")
    print("Learning rate: ",lr)

    model = MyAwesomeModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use cuda or cpu
    print("Used device: ", device)
    model.to(device)

    batchsize = 64
    num_epochs = 10

    train_set, _ = mnist()
        
    train_loader = DataLoader(train_set,batch_size=batchsize,shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
    
        model.train()
        print("Epoch number: " + str(epoch))

        train_losses = []

        for inputs, targets in enumerate(train_loader,0):
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            output = model(inputs)
            print("output", output)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        print("train_losses: ", train_losses)

        # print('Training Loss: ' + str(np.mean(np.array(train_losses))))


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    