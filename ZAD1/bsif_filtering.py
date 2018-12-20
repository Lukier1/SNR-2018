import os

import cv2
import scipy.io as sio
import numpy as np
from pathlib import Path
import csv
from utils import output_root

images_scan_root = "images/Folio"
bsif_path = "bsif_filters/"


def get_filter_name(size, depth):
    return f"ICAtextureFilters_{size}x{size}_{depth}bit.mat"


def image_on_bsif_filter(img, filter_size, filter_depth):
    filter_name = get_filter_name(filter_size, filter_depth)
    filter_path = os.path.join(bsif_path, filter_name)
    filter_ = sio.loadmat(filter_path)['ICAtextureFilters']
    out_img = np.zeros((img.shape[0], img.shape[1]))
    print(filter_depth)
    for i in range(0, filter_depth):
        kernel = filter_[:, :, i]
        dst = cv2.filter2D(img, -1, kernel)
        ret, thresh1 = cv2.threshold(dst, 1, 1, cv2.THRESH_BINARY)
        out_img[:, :] += thresh1 * (1 << i)
    return out_img


def load_gray_image(file_name):
    img = cv2.imread(file_name)
    # Tutaj byla analiza po kolorach
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(img, (30,0,0), (120,255,255))
    # imask = mask>0
    # green = np.zeros_like(img, np.uint8)
    # green[imask] = img[imask]
    # return green[:, :, 1]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def proceed_images(filter_size, filter_depth):
    p = Path(images_scan_root)
    label_counter = {}
    label_id = {}
    idx = 0

    print(f"Start processing size: {filter_size} bits: {filter_depth}")

    for child in p.glob("**/*"):
        if not child.is_dir():
            label = str(child.parent)
            label = label.replace(images_scan_root + "/", "")
            if label in label_counter:
                label_counter[label] += 1
            else:
                label_counter[label] = 0

            filename = label + str(label_counter[label]) + ".npy"
            label_id[filename] = idx
            idx += 1

            img = load_gray_image(str(child))
            img = cv2.resize(img, (73, 129))
            out = image_on_bsif_filter(img, filter_size, filter_depth)

            cv2.imwrite(
                output_root(filter_size, filter_depth) + "/" + label + str(
                    label_counter[label]) + ".png", out)
            np.save(output_root(filter_size, filter_depth) + "/" + filename,
                    out)

    with open(output_root(filter_size, filter_depth) + '/index.csv',
              'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)
        for key in label_id:
            writer.writerow([key, label_id[key]])
    print(f"Saved size: {filter_size} bits: {filter_depth}")
