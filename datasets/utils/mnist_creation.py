import os
from itertools import product
from pathlib import Path
from random import sample, choice
from torchvision import transforms
import numpy as np
import torch
from torch import tensor, load
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm
import copy





class nMNIST(Dataset):
    """nMNIST dataset."""

    def __init__(self, train=False, data_path=None):
        print('Loading data...')
        self.data, self.labels = self.read_data(path=data_path, train=train)
        self.targets= self.labels[:,-1:].reshape(-1)
        self.concepts= self.labels[:,:-1]
        self.real_concepts = np.copy(self.labels[:,:-1])
        normalize = transforms.Normalize((0.1307,), (0.3081,))

        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # implicitly divides by 255
        normalize
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
            image = self.data[idx]
            label=self.targets[idx]
            concepts=self.concepts[idx]

            if self.transform is not None:
                newimage=self.transform(image.astype("uint8"))

            if hasattr(self, 'logits'):
                return newimage, label,concepts,image.astype("uint8"), self.logits[idx]

            return newimage, label,concepts,image.astype("uint8")

    def read_data(self, path, train=True):
        """
        Returns images and labels
        """
        try:
            # print("Loading data...")
            data = load(path)
            # print("Loaded.")
        except:
            print("No dataset found.")

        if train:
            images = data['train']['images']
            labels = data['train']['labels']
        else:
            images = data['test']['images']
            labels = data['test']['labels']

        return images, labels

    def reset_counter(self):
        self.world_counter = {c: self.samples_x_world // self.batch_size for c in self.worlds}
        # self.world_counter = {c: self.samples_x_world for c in self.worlds}

    # def __copy__():

def create_sample(X, target_sequence, digit2idx):
    idxs = [choice(digit2idx[digit][0]) for digit in target_sequence]
    imgs = [X[idx] for idx in idxs]
    new_image = np.concatenate(imgs, axis=1)
    new_label = target_sequence + (np.sum(target_sequence),)

    return new_image, np.array(new_label).astype('int32'), idxs


def create_dataset(n_digit=2, sequence_len=2, samples_x_world=100, train=True, download=False):
    # Download data
    MNIST = datasets.MNIST(root='./data/raw/', train=train, download=download)

    x, y = MNIST.data, MNIST.targets

    # Create dictionary of indexes for each digit
    digit2idx = {k: [] for k in range(10)}
    for k, v in digit2idx.items():
        v.append(np.where(y == k)[0])

    # Create the list of all possible permutations with repetition of 'sequence_len' digits
    worlds = list(product(range(n_digit), repeat=sequence_len))
    imgs = []
    labels = []

    # Create data sample for each class
    for c in tqdm(worlds):
        for i in range(samples_x_world):
            img, label, idxs = create_sample(x, c, digit2idx)
            imgs.append(img)
            labels.append(label)

    # Create dictionary of indexes for each world
    label2idx = {c: set() for c in worlds}
    for k, v in tqdm(label2idx.items()):
        for i, label in enumerate(labels):
            if tuple(label[:2]) == k:
                v.add(i)
    label2idx = {k: tensor(list(v)) for k, v in label2idx.items()}


    return np.array(imgs).astype('int32'), np.array(labels), label2idx


def check_dataset(n_digits, data_folder, data_file, dataset_dim):
    """Checks whether the dataset exists, if not creates it."""
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    data_path = os.path.join(data_folder, data_file)
    try:
        load(data_path)
    except:
        print("No dataset found.")
        # Define dataset dimension so to have teh same number of worlds
        n_worlds = n_digits * n_digits
        samples_x_world = {k: int(d / n_worlds) for k, d in dataset_dim.items()}
        dataset_dim = {k: s * n_worlds for k, s in samples_x_world.items()}

        train_imgs, train_labels, train_indexes = create_dataset(n_digit=n_digits, sequence_len=2,
                                                                 samples_x_world=samples_x_world['train'], train=True,
                                                                 download=True)

        val_imgs, val_labels, val_indexes = create_dataset(n_digit=n_digits, sequence_len=2,
                                                           samples_x_world=samples_x_world['val'], train=True,
                                                           download=True)
        test_imgs, test_labels, test_indexes = create_dataset(n_digit=n_digits, sequence_len=2,
                                                              samples_x_world=samples_x_world['test'], train=False,
                                                              download=True)


        print(
            f"Dataset dimensions: \n\t{dataset_dim['train']} train ({samples_x_world['train']} samples per world), \n\t{dataset_dim['val']} validation ({samples_x_world['val']} samples per world), \n\t{dataset_dim['test']} test ({samples_x_world['test']} samples per world)")

        data = {'train': {'images': train_imgs, 'labels': train_labels},
                'val': {'images': val_imgs, 'labels': val_labels},
                'test': {'images': test_imgs, 'labels': test_labels}}

        indexes = {'train': train_indexes,
                   'val': val_indexes,
                   'test': test_indexes}

        torch.save(data, data_path)
        for key, value in indexes.items():
            torch.save(value, os.path.join(data_folder, f'{key}_indexes.pt'))

        print(f"Dataset saved in {data_folder}")

def load_2MNIST(batch_size, n_digits=10,
                  dataset_dimensions= {'train': 42000, 'val': 12000, 'test': 6000},
                  c_sup=1):

    # Load data
    data_folder = os.path.dirname(os.path.abspath(__file__))

    # Load data
    data_file =  f'2mnist_{n_digits}digits.pt'
    data_folder = os.path.join(data_folder, f'2mnist_{n_digits}digits')
    # Check whether dataset exists, if not build it

    batch_size = {'train': batch_size, 'val': batch_size, 'test': batch_size, }
    check_dataset(n_digits, data_folder, data_file, dataset_dimensions)
    train_set, val_set, test_set = load_data(data_file=data_file, data_folder=data_folder, task='base',
                                            c_sup=c_sup)

    return train_set, val_set, test_set


def load_data(data_file, data_folder, task='base', c_sup=1):
    if task == 'base':
        # Prepare data
        data_path = os.path.join(data_folder, data_file)
        train_set = nMNIST(train=True,  data_path=data_path)

        train_set, val_set = shuffle_train_val(train_set, 0.8)
        test_set  = nMNIST(train=False,  data_path=data_path)

        if c_sup < 1:
            # select examples to remove supervision
            mask = np.random.rand(len(train_set.concepts)) > c_sup
            train_set.concepts[mask] = -1

    return train_set, val_set, test_set

def shuffle_train_val(train: nMNIST, c=0.8):


    train_set = copy.deepcopy(train)
    val_set   = copy.deepcopy(train)

    perm = torch.randperm(len(train_set))

    data = train.data[perm]
    labels = train.targets[perm]
    concepts = train.concepts[perm]
    real_concepts = train.real_concepts[perm]

    sep = round(c * len(data))
    train_set.data, val_set.data = data[:sep], data[sep:]
    train_set.targets, val_set.targets = labels[:sep], labels[sep:]
    train_set.concepts, val_set.concepts = concepts[:sep], concepts[sep:]
    train_set.real_concepts, val_set.real_concepts = real_concepts[:sep], real_concepts[sep:]
    print('Len train', len(train_set))
    print('Len val', len(val_set))

    return train_set, val_set