import numpy as np
import torch
import csv
import re

from torch.utils import data

from utils import output_root


class Dataset(data.Dataset):

    def __init__(self, filter_size, filter_depth):
        self.id_label = {}
        self.id_filename = {}
        self.filter_size = filter_size
        self.filter_depth = filter_depth
        with open(output_root(filter_size,
                              filter_depth) + "/" + 'index.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.id_filename[row[1]] = row[0]
                label = self.get_label(row[0])
                if not label in self.id_label:
                    self.id_label[label] = len(self.id_label)

    def __len__(self):
        return len(self.id_filename)

    def __getitem__(self, index):
        filename = self.id_filename[str(index)]
        label = self.get_label(filename)
        label = int(self.id_label[label])

        filtered_img = np.load(
            output_root(self.filter_size, self.filter_depth) + "/" + filename)
        filtered_img /= (1 << self.filter_depth)
        features = torch.from_numpy(filtered_img).float()
        # features = features.type(torch.FloatStorage)
        return features, label

    def get_label(self, filename):
        return re.split(r'(\D*)(\d+)\.npy', filename)[1]

    def get(self, index):
        label = self.get_label(self.id_filename[str(index)])
        label = int(self.id_label[label])
        return self.__getitem__(index)

    def get_label_name(self, search_id):
        for label, id in self.id_label.items():
            if id == search_id:
                return label
        return ""
