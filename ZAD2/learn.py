from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models

DATA_DIR = 'images/Folio'
VALIDATION_SPLIT = .2
SEED = 42  # fixed seed to have repeatable results
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learn_and_save_model(augment=False, shuffle=True, train_whole=False):
    train_loader, valid_loader, test_loader = get_data_loaders(augment, shuffle)

    model = get_model(train_whole)

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    epochs = 10
    steps = 0
    running_loss = 0
    print_every = 1
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Step {steps}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_loader):.3f}")
            running_loss = 0
            model.train()

    torch.save(model.state_dict(), f"output_{train_whole}")


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


def get_model(train_whole):
    model = models.densenet121(True)
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
    learn_and_save_model(train_whole=True)
    learn_and_save_model(train_whole=False)
