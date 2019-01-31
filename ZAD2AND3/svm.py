from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import torch
from sklearn import svm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from cnn import device, get_model, plot_cmc_curve, save_topk_accuracy, \
    create_output_dir, get_transforms, TRAIN_DATA_DIR, TEST_DATA_DIR,\
    BATCH_SIZE


def get_data_loaders():
    train_transforms, test_transforms = get_transforms(False)
    train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR,
                                         transform=train_transforms)
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR,
                                        transform=test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train_loader, test_loader


def extract_features_with_cnn(cnn_path):
    print(f"Extracting features using CNN: {cnn_path}")
    train_loader, test_loader = get_data_loaders()
    # load pytorch NN and use as feature extractor
    feature_extractor = get_model('densenet121', True)
    feature_extractor.load_state_dict(torch.load(cnn_path))
    feature_extractor.classifier = nn.Sequential()
    feature_extractor.to(device)
    feature_extractor.eval()
    Xs_train, ys_train, Xs_test, ys_test = [], [], [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = feature_extractor.forward(inputs)
            Xs_train.append(features)
            ys_train.append(labels)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = feature_extractor.forward(inputs)
            Xs_test.append(features)
            ys_test.append(labels)
    X_train = torch.cat(Xs_train).to('cpu').numpy()
    y_train = torch.cat(ys_train).to('cpu').numpy()
    X_test = torch.cat(Xs_test).to('cpu').numpy()
    y_test = torch.cat(ys_test).to('cpu').numpy()
    return X_train, y_train, X_test, y_test


def fit_svm(X_train, y_train, X_test, y_test, poly_degree, output_dir,
            cnn_path):
    file_prefix = f"svm_poly_{poly_degree}_{cnn_path.name[:-3]}"

    classifier = svm.SVC(kernel='poly', gamma='auto', degree=poly_degree,
                         probability=True, verbose=False)
    classifier.fit(X_train, y_train)
    y_probas = classifier.predict_proba(X_test)

    top_class = np.argsort(y_probas, axis=1)[:, ::-1]
    top_correct = top_class == y_test[:, None]
    ranks = top_correct.argmax(1) + 1
    ranks_counter = Counter({rank: 0 for rank in range(33)})
    ranks_counter.update(ranks)
    ranks_counts = np.asarray(sorted(ranks_counter.most_common()))
    topk_accuracy = np.cumsum(ranks_counts[:, 1]) / len(y_test)

    save_topk_accuracy(topk_accuracy, file_prefix, output_dir)
    plot_cmc_curve(topk_accuracy, file_prefix, output_dir)

    roc_title = f"ROC Curves - {poly_degree} degree polynominal SVM kernel"
    skplt.metrics.plot_roc(y_test, y_probas, title=roc_title, plot_macro=False,
                           plot_micro=False, ncol=2, figsize=(12, 9))
    plt.savefig(Path(output_dir, f"{file_prefix}_roc.png"), dpi=300)
    plt.close()


def run():
    output_dir = create_output_dir()
    cnn_path = Path("output",
                    "2019-01-31T12_04_31_471780",
                    "model_densenet121_classifier_training_no_augmentation_Adam_final.pt")
    X_train, y_train, X_test, y_test = extract_features_with_cnn(cnn_path)
    for poly_degree in range(1, 6):
        fit_svm(X_train, y_train, X_test, y_test, poly_degree, output_dir,
                cnn_path)


if __name__ == '__main__':
    run()
