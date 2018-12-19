import csv
import math
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms

from dataset import Dataset
from model import Classifier
from utils import output_root, model_output_filename


CLASS_NUMBER = 32

def calcStatAndGenPlotForModel(filter_size, filter_depth, layer):
    model = Classifier(layer)

    model_filename = model_output_filename(filter_size, filter_depth, layer)
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    dataset = Dataset(filter_size, filter_depth)

    correct_top1 = 0
    correct_top5 = 0
    rank_array = np.zeros((CLASS_NUMBER)) 

    t_c = np.zeros((CLASS_NUMBER, 102))
    f_c = np.zeros((CLASS_NUMBER, 102))
    t_n = np.zeros((CLASS_NUMBER, 1))
    f_n = np.zeros((CLASS_NUMBER, 1))

    for i in range(0, dataset.__len__()):
        features, labels = dataset.get(i)
        features_2 = features.view(1, features.shape[0]*features.shape[1])


        logps = model.forward(features_2)
        
        chance_log, predicted = torch.max(logps, 1)
        label_id = labels 
        predicted_chance = math.exp(chance_log[0])
        chance_table = torch.exp(logps)

        #rank 
        class_prob = logps.data[0][label_id]
        rank = 0
        for x in range(0, logps.shape[1]):
            if logps.data[0][x] > class_prob:
                rank += 1
        
        #Top-1 Calculating
        if rank == 0:
            correct_top1 += 1

        #Top-5 Calculalting
        if rank < 5:
            correct_top5 += 1
        
        #CMC
        for x in range(rank, CLASS_NUMBER):
            rank_array[x] = rank_array[x] + 1
        
        chance_idx = math.ceil(predicted_chance*100)
            
        #ROC
        for y in range(0, CLASS_NUMBER):
            chance_idx = math.ceil(chance_table[0, y]*100)+1
            if y == label_id: #klasa pozytywna(P)
                t_c[y, :chance_idx ] += 1
                t_n[y, 0] += 1
            else: #klasa negatywna
                f_c[y, :chance_idx ] += 1
                f_n[y, 0] += 1
                
        
    print("Top 1 = " + str(correct_top1/dataset.__len__()))
    print("Top 5 = " + str(correct_top5/dataset.__len__()))

    x = np.arange(CLASS_NUMBER)
    y = rank_array[x]/dataset.__len__()
    fig, ax = plt.subplots()

    # Using set_dashes() to modify dashing of an existing line
    line1, = ax.plot(x, y, label=f'CMC for {filter_size}x{filter_size} {filter_depth}bit')
    #line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

    # Using plot(..., dashes=...) to set the dashing when creating a line
    #line2, = ax.plot(x, y - 0.2, dashes=[6, 2], label='Using the dashes parameter')

    ax.legend()
    plt.show()

    tpr = t_c/t_n
    fpr = f_c/f_n

    y2 = tpr[:, :]
    x2 = fpr[:, :]


    fig2, ax2 = plt.subplots()

    # Using set_dashes() to modify dashing of an existing line
    for y in range(0, CLASS_NUMBER):
        _, = ax2.plot(x2[y, :], y2[y, :], label=dataset.getLabelName(y))
    #line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

    # Using plot(..., dashes=...) to set the dashing when creating a line
    #line2, = ax.plot(x, y - 0.2, dashes=[6, 2], label='Using the dashes parameter')

    ax2.legend()
    plt.show()