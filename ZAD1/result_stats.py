import csv
import math

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

OUTPUT_STATS = "stats/"
CLASS_NUMBER = 32


def calc_stat_and_gen_plot_for_model(filter_size, filter_depth, layer):
    print(f'Calc stats for {filter_size}x{filter_size} {filter_depth}'
          f'bits {layer} layer')
    model = Classifier(layer)

    model_filename = model_output_filename(filter_size, filter_depth, layer)
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    dataset = Dataset(filter_size, filter_depth)

    correct_top1 = 0
    correct_top5 = 0
    rank_array = np.zeros(CLASS_NUMBER)

    t_c = np.zeros((CLASS_NUMBER, 102))
    f_c = np.zeros((CLASS_NUMBER, 102))
    tn_c = np.zeros((CLASS_NUMBER, 102))
    tp_c = np.zeros((CLASS_NUMBER, 102))
    t_n = np.zeros((CLASS_NUMBER, 1))
    f_n = np.zeros((CLASS_NUMBER, 1))

    for i in range(0, dataset.__len__()):
        features, labels = dataset.get(i)
        features_2 = features.view(1, features.shape[0] * features.shape[1])

        logps = model.forward(features_2)

        chance_log, predicted = torch.max(logps, 1)
        label_id = labels
        predicted_chance = math.exp(chance_log[0])
        chance_table = torch.exp(logps)

        # rank
        class_prob = logps.data[0][label_id]
        rank = 0
        for x in range(0, logps.shape[1]):
            if logps.data[0][x] > class_prob:
                rank += 1

        # Top-1 Calculating
        if rank == 0:
            correct_top1 += 1

        # Top-5 Calculalting
        if rank < 5:
            correct_top5 += 1

        # CMC
        for x in range(rank, CLASS_NUMBER):
            rank_array[x] = rank_array[x] + 1

        chance_idx = math.ceil(predicted_chance * 100)

        # ROC
        for y in range(0, CLASS_NUMBER):
            chance_idx = math.ceil(chance_table[0, y] * 100) + 1
            if y == label_id:  # klasa pozytywna(P)
                t_c[y, :chance_idx] += 1
                if y == predicted[0]:
                    tp_c[y, 0] += 1
                t_n[y, 0] += 1
            else:  # klasa negatywna
                f_c[y, :chance_idx] += 1
                f_n[y, 0] += 1
                if y != predicted[0]:
                    tn_c[y, 0] += 1

    print("Top 1 = " + str(correct_top1 / dataset.__len__()))
    print("Top 5 = " + str(correct_top5 / dataset.__len__()))

    filename_part = (f'Result for {filter_size}x{filter_size} {filter_depth}'
                     f'bit {layer} layers')

    with open(OUTPUT_STATS + filename_part + 'top.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)
        writer.writerow(['TOP-1', str(correct_top1 / dataset.__len__())])
        writer.writerow(['TOP-5', str(correct_top5 / dataset.__len__())])
        writer.writerow(['Class', 'Accurancy'])

        # Accuracy ACC = TP+TN/ALL
        for y in range(0, CLASS_NUMBER):
            acc = (tp_c[y, 0] + tn_c[y, 0]) / (t_n[y, 0] + f_n[y, 0])
            writer.writerow([dataset.get_label_name(y), acc])

    x = np.arange(CLASS_NUMBER)
    y = rank_array[x] / dataset.__len__()
    fig, ax = plt.subplots()
    label = (f'CMC for {filter_size}x{filter_size} {filter_depth}'
             f'bit {layer} layers')
    line1, = ax.plot(x, y, label=label)
    ax.legend()
    plt.savefig(OUTPUT_STATS + filename_part + "_CMC.png", bbox_inches="tight")

    tpr = t_c / t_n
    fpr = f_c / f_n

    y2 = tpr[:, :]
    x2 = fpr[:, :]

    fig2, ax2 = plt.subplots()
    plt.subplots_adjust(right=0.7)
    labels_legend = []
    for y in range(0, CLASS_NUMBER):
        _, = ax2.plot(x2[y, :], y2[y, :], label=dataset.get_label_name(y))
    plt.title(f"{filter_size}x{filter_size} {filter_depth}bit {layer} layers")
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    plt.savefig(OUTPUT_STATS + filename_part + "_ROC.png", bbox_inches="tight")
    print('... Finish calc stats')
# calcStatAndGenPlotForModel(15, 8, 5)
