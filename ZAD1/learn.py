import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from dataset import Dataset
from model import Classifier
from utils import model_output_filename, output_root


def learnAndSaveModel(filter_size, filter_depth, layer_num):
    #data set properties
    batch_size = 100
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    #load flat and normalize image
    dataset = Dataset(filter_size, filter_depth)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    #Loading datasets train, validation, test
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
   
    #Loading model, and optimizer
    model = Classifier(layer_num)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.NLLLoss()

    #Stop conition values    
    last_min = 300
    last_min_idx = -1

    #Learning loop
    max_num_epochs = 20000
    for x in range(0, max_num_epochs):
        running_loss = 0
        for images, labels in train_loader:
            features = images.view(images.shape[0], -1)
            logps = model.forward(features)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
        with torch.set_grad_enabled(False):
            for images, labels in validation_loader:
                # Load image
                features = images.view(images.shape[0], -1)
                
                # Model computations
                logps = model.forward(features)
                loss = criterion(logps, labels)
                running_loss += loss.item()
                if loss.item() < last_min:
                    last_min = loss.item()
                    last_min_idx = x
            else:
                print (f"Running loss: {running_loss/len(validation_loader)}")
        
        #Stop condition
        if x-last_min_idx > 15:
            print(f'Epochs {x}')
            break;

    #saving model
    torch.save(model.state_dict(), model_output_filename(filter_size, filter_depth, layer_num))

    running_loss = 0
    for images, labels in test_loader:
        # Transfer to GPU
        features = images.view(images.shape[0], -1)
        
        # Model computations
        logps = model.forward(features)
        loss = criterion(logps, labels)
        running_loss += loss.item()
    else:
        print (f"Final cost is: {running_loss/len(test_loader)}")

