import csv
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models

from pytorchtools import EarlyStopping

DATA_DIR = 'images/Folio'
VALIDATION_SPLIT = .2
SEED = 42  # fixed seed to have repeatable results
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")


def learn_and_save_model(augment=False, train_whole=False, max_epoches=100):
    model_to_train = "densenet121"
    print(f"Training {model_to_train}.. Training whole model: {train_whole}.. "
          f"Augment: {augment}.. Max epoches: {max_epoches}..")
    train_loader, valid_loader, test_loader = get_data_loaders(augment, True)
    model = get_model(model_to_train, train_whole)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    file_prefix = f"model_{model_to_train}_{train_whole}_{augment}"
    train_model(model, train_loader, valid_loader, optimizer, criterion,
                max_epoches, file_prefix)


def train_model(model, train_loader, valid_loader, optimizer, criterion,
                max_epochs, prefix):
    model.to(device)
    train_loss_history = []
    valid_loss_history = []
    accuracy_history = []
    time_history = []
    early_stopping = EarlyStopping(verbose=True, patience=5,
                                   f=f"{prefix}_checkpoint.pt")
    for epoch in range(max_epochs):
        ts = time.time()
        train_losses = []
        valid_losses = []
        accuracies = []
        # Train
        model.train()
        for inputs, labels in train_loader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        # Evaluate
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model.forward(inputs)
                loss = criterion(output, labels)
                valid_losses.append(loss.item())
                accuracies.append(calculate_accuracy(labels, output))
        te = time.time()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        accuracy = np.average(accuracies)
        tt = (te - ts) * 1000
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        accuracy_history.append(accuracy)
        time_history.append(tt)
        print(f"Epoch {epoch+1}/{max_epochs}.. "
              f"Took {tt}ms.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Validation loss: {valid_loss:.3f}.. "
              f"Validation accuracy: {accuracy:.3f}")
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print(f"Finished training in {sum(time_history)}")
    early_stopping.load_checkpoint(model)
    torch.save(model.state_dict(), f"{prefix}_final.pt")
    save_training_history(accuracy_history, train_loss_history,
                          valid_loss_history, time_history)


def save_training_history(*histories):
    with open(f"{prefix}_training_history.csv", 'w') as csv_file:
        writer = csv.writer(csv_file, delimeter=' ')
        for row in zip(*histories):
            writer.writerow(row)


def calculate_accuracy(labels, logps):
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
    return accuracy


def get_data_loaders(augment, shuffle):
    train_transforms, valid_transforms = get_transforms(augment)
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(DATA_DIR, transform=valid_transforms)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    if shuffle:
        np.random.seed(SEED)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                              sampler=valid_sampler)
    # TODO: jkumor - for now it uses all data - not sure if this is correct approach
    test_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    return train_loader, valid_loader, test_loader


def get_transforms(augment):
    common_transforms = [transforms.Resize(224),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])]
    if augment:
        transform_list = ([transforms.RandomRotation(30),
                           transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip()]
                          + common_transforms[2:])
        train_transforms = transforms.Compose(transform_list)
    else:
        train_transforms = transforms.Compose(common_transforms)
    valid_transforms = transforms.Compose(common_transforms)
    return train_transforms, valid_transforms


def get_model(name, train_whole):
    model = getattr(models, name)(pretrained=True)
    if not train_whole:
        # Freeze parameters so we don't do back propagation through them
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict(fc1=nn.Linear(1024, 500),
                                                 relu=nn.ReLU(),
                                                 fc2=nn.Linear(500, 32),
                                                 output=nn.LogSoftmax(dim=1)))
    return model


if __name__ == '__main__':
    learn_and_save_model(train_whole=False)
    learn_and_save_model(train_whole=True)
    learn_and_save_model(augment=True, train_whole=True)
