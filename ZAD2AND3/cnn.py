import csv
import datetime
import time
from collections import OrderedDict, Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models


from pytorchtools import EarlyStopping

plt.style.use('ggplot')
TRAIN_DATA_DIR = 'images_separated/folio_train'
TEST_DATA_DIR = 'images_separated/folio_test'
VALIDATION_SPLIT = .25
SEED = 42  # fixed seed to have repeatable results
BATCH_SIZE = 16
CLASS_NUMBER = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")


def no_train_and_save(output_dir):
    model_to_train = "densenet121"
    prefix = f"model_{model_to_train}_no_train"
    model = get_model(model_to_train, True)
    _, _, test_loader = get_data_loaders(False, False)
    torch.save(model.state_dict(), Path(output_dir, f"{prefix}_final.pt"))
    generate_stats_and_plots(model, test_loader, prefix, output_dir)


def train_and_save_model(output_dir, augment=False, train_whole=False,
                         drop_rate=0, max_epoches=10, do_early_stop=False):
    model_to_train = "densenet121"
    print(f"Training {model_to_train}.. Training whole model: {train_whole}.. "
          f"Augment: {augment}.. Max epoches: {max_epoches}..")
    train_loader, valid_loader, test_loader = get_data_loaders(augment, True)
    model = get_model(model_to_train, train_whole, drop_rate)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()
    file_prefix = get_file_prefix(model_to_train, train_whole, augment,
                                  drop_rate, optimizer)
    train_model(model, train_loader, valid_loader, optimizer, criterion,
                max_epoches, file_prefix, output_dir, do_early_stop)
    generate_stats_and_plots(model, test_loader, file_prefix, output_dir)


def create_output_dir():
    """Creates output directory."""
    date = datetime.datetime.now().isoformat()
    path = Path("output", date.replace(':', '_').replace('.', '_'))
    path.mkdir(parents=True, exist_ok=False)
    return path


def get_file_prefix(model_to_train, train_whole, augment, drop_rate, optimizer):
    prefix = f"model_{model_to_train}"
    prefix += "_whole_training" if train_whole else "_classifier_training"
    prefix += "_with_augmentation" if augment else "_no_augmentation"
    prefix += f"_drop_rate_{drop_rate*100}"
    prefix += f"_{type(optimizer).__name__}"
    return prefix


def train_model(model, train_loader, valid_loader, optimizer, criterion,
                max_epochs, prefix, output_dir, do_early_stop=False):
    model.to(device)
    train_loss_history = []
    valid_loss_history = []
    accuracy_history = []
    time_history = []
    early_stopping = EarlyStopping(verbose=True, patience=3,
                                   f=Path(output_dir,
                                          f"{prefix}_checkpoint.pt"),
                                   dummy=not do_early_stop)
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
        # Validate
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
        tt = round((te - ts) * 1000)
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
        if do_early_stop and early_stopping.early_stop:
            print("Early stopping")
            break
    print(f"Finished training in {sum(time_history)}ms")
    early_stopping.load_checkpoint(model)
    torch.save(model.state_dict(), Path(output_dir, f"{prefix}_final.pt"))
    save_training_history(output_dir, prefix, train_loss_history,
                          valid_loss_history, accuracy_history, time_history)
    plot_loss_history(output_dir, prefix, train_loss_history,
                      valid_loss_history)


def plot_loss_history(output_dir, prefix, train_loss_history,
                      valid_loss_history):
    fig = plt.figure()
    plt.plot(train_loss_history, label=f'Training loss')
    plt.plot(valid_loss_history, label=f'Validation loss')
    plt.title("Training loss history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(output_dir, f"{prefix}_loss.png"), bbox_inches="tight")
    plt.close(fig)


def save_training_history(output_dir, prefix, *histories):
    path = Path(output_dir, f"{prefix}_training_history.csv")
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
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
    train_dataset = datasets.ImageFolder(TRAIN_DATA_DIR,
                                         transform=train_transforms)
    valid_dataset = datasets.ImageFolder(TRAIN_DATA_DIR,
                                         transform=valid_transforms)
    test_dataset = datasets.ImageFolder(TEST_DATA_DIR,
                                        transform=valid_transforms)
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
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train_loader, valid_loader, test_loader


def get_transforms(augment, obscure=None):
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


def get_model(name, train_whole, drop_rate=0):
    model = getattr(models, name)(pretrained=True, drop_rate=drop_rate)
    if not train_whole:
        # Freeze parameters so we don't do back propagation through them
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict(fc1=nn.Linear(1024, 500),
                                                 relu=nn.ReLU(),
                                                 fc2=nn.Linear(500,
                                                               CLASS_NUMBER),
                                                 output=nn.LogSoftmax(dim=1)))
    return model


def generate_stats_and_plots(model, test_loader, prefix, output_dir):
    print(f'Calculating statistics')

    model.eval()
    model.to(device)
    ranks_count = Counter({rank: 0 for rank in range(CLASS_NUMBER+1)})
    y_trues = []
    y_probass = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_ps, top_class = ps.sort(dim=1, descending=True)
            top_correct = top_class.eq(labels.view(-1, 1).expand_as(top_class))
            ranks = top_correct.nonzero()[:, 1] + 1
            ranks_count.update(ranks.tolist())
            y_trues.append(labels)
            y_probass.append(ps)
    y_true = torch.cat(y_trues).to('cpu').numpy()
    y_probas = torch.cat(y_probass).to('cpu').numpy()

    ranks_count_np = np.asarray(sorted(ranks_count.most_common()))[:, 1]
    topk_accuracy = np.cumsum(ranks_count_np) / len(test_loader.dataset)
    save_topk_accuracy(topk_accuracy, prefix, output_dir)
    plot_cmc_curve(topk_accuracy, prefix, output_dir)
    plot_roc_curve(y_probas, y_true, prefix, output_dir)


def plot_roc_curve(y_probas, y_true, prefix, output_dir):
    roc_title = f"ROC Curves"
    skplt.metrics.plot_roc(y_true, y_probas, title=roc_title, plot_macro=False,
                           plot_micro=False, ncol=2, figsize=(12, 9))
    plt.savefig(Path(output_dir, f"{prefix}_roc.png"), dpi=300)


def save_topk_accuracy(topk_accuracy, prefix, output_dir):
    path = Path(output_dir, f"{prefix}_topk_accuracy.csv")
    with open(path, "w", newline='') as topk_file:
        writer = csv.writer(topk_file, delimiter=",")
        writer.writerow(["k", "top-k"])
        for n, topn_accuracy in enumerate(topk_accuracy):
            writer.writerow([n, topn_accuracy])
    top1_accuracy = topk_accuracy[1]
    top5_accuracy = topk_accuracy[5]
    print(f"Top 1 = {top1_accuracy}")
    print(f"Top 5 = {top5_accuracy}")


def plot_cmc_curve(topk_accuracy, prefix, output_dir):
    fig = plt.figure()
    x = np.arange(CLASS_NUMBER+1)
    y = topk_accuracy
    plt.plot(x, y, label=f'CMC')
    plt.plot([0, max(x)], [0, 1], linestyle='--', label='chance')
    plt.title("Cumulative matching characteristic")
    plt.xlabel("Rank")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(Path(output_dir, f"{prefix}_cmc.png"), bbox_inches="tight")
    plt.close(fig)


def run():
    output_dir = create_output_dir()
    no_train_and_save(output_dir)
    train_and_save_model(output_dir, train_whole=False, augment=False,
                         max_epoches=20)
    train_and_save_model(output_dir, train_whole=False, augment=False,
                         max_epoches=20, do_early_stop=True)
    train_and_save_model(output_dir, train_whole=False, augment=False,
                         max_epoches=10)


if __name__ == '__main__':
    run()
