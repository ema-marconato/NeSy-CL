import numpy as np
import torch
from torchvision import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.utils.mnist_creation import load_2MNIST
from datasets.utils.continual_dataset import ContinualDataset
from backbone.MNIST_DeepProblog import MNIST_DeepProblog
from backbone.MNIST_CBM import MNIST_CBM
from backbone.MNIST_baseline import MNIST_baseline
import numpy as np
from typing import Tuple


class ShortcutMNIST(ContinualDataset):

    NAME = 'shortcut-mnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK =3
    N_CONCEPTS_PER_TASK=5
    N_TASKS = 2
    TRANSFORM = None
    SEQUENCE_LEN=2
    n_digits=10
    batch_size=30

    def get_data_loaders(self, full=False):
        train_dataset,val_dataset,test_dataset=load_2MNIST(self.args.batch_size, c_sup=self.args.c_sup)

        if not full:
            if not self.args.val_search:
                train, test = store_masked_loaders(train_dataset, test_dataset, self)
                return train, test
            else:
                train, val = store_masked_loaders(train_dataset, val_dataset, self)
                return train, val
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)# num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)# num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)# num_workers=4)
            return train_loader, val_loader, test_loader

    @staticmethod
    def get_backbone(version):
        if version == 'nesy':
            return MNIST_DeepProblog()
        elif version == 'cbm':
            return MNIST_CBM()
        elif version=='baseline':
            return MNIST_baseline()
        else:
            return None

    @staticmethod
    def get_transform():
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # implicitly divides by 255
        normalize])
        return transform

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        model.opt = torch.optim.Adam(model.net.parameters(), lr=args.lr) #, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(model.opt, 0.95)
        return scheduler

def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset, task='standard') -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """

    # print('len train', len(train_dataset))

    ## CLASS-INCREMENTAL
    print(type(train_dataset.real_concepts))
    print(train_dataset.real_concepts.shape)
    train_dataset.concepts[:,0] == 0
    (np.array(train_dataset.concepts[:,0]) == 0) | (np.array(train_dataset.concepts[:,0]) == 2 )

    if setting.i == 0:
        train_c_mask1 = ((train_dataset.real_concepts[:,0] == 0) & (train_dataset.real_concepts[:,1] == 6)) | \
                        ((train_dataset.real_concepts[:,0] == 2) & (train_dataset.real_concepts[:,1] == 8)) | \
                        ((train_dataset.real_concepts[:,0] == 4) & (train_dataset.real_concepts[:,1] == 6)) | \
                        ((train_dataset.real_concepts[:,0] == 4) & (train_dataset.real_concepts[:,1] == 8))
        train_c_mask2 = ((train_dataset.real_concepts[:,1] == 0) & (train_dataset.real_concepts[:,0] == 6)) | \
                        ((train_dataset.real_concepts[:,1] == 2) & (train_dataset.real_concepts[:,0] == 8)) | \
                        ((train_dataset.real_concepts[:,1] == 4) & (train_dataset.real_concepts[:,0] == 6)) | \
                        ((train_dataset.real_concepts[:,1] == 4) & (train_dataset.real_concepts[:,0] == 8))
        train_mask = np.logical_or(train_c_mask1, train_c_mask2)

        test_c_mask1 = ((test_dataset.real_concepts[:,0] == 0) & (test_dataset.real_concepts[:,1] == 6)) | \
                        ((test_dataset.real_concepts[:,0] == 2) & (test_dataset.real_concepts[:,1] == 8)) | \
                        ((test_dataset.real_concepts[:,0] == 4) & (test_dataset.real_concepts[:,1] == 6)) | \
                        ((test_dataset.real_concepts[:,0] == 4) & (test_dataset.real_concepts[:,1] == 8))
        test_c_mask2 = ((test_dataset.real_concepts[:,1] == 0) & (test_dataset.real_concepts[:,0] == 6)) | \
                        ((test_dataset.real_concepts[:,1] == 2) & (test_dataset.real_concepts[:,0] == 8)) | \
                        ((test_dataset.real_concepts[:,1] == 4) & (test_dataset.real_concepts[:,0] == 6)) | \
                        ((test_dataset.real_concepts[:,1] == 4) & (test_dataset.real_concepts[:,0] == 8))
        test_mask = np.logical_or(test_c_mask1, test_c_mask2)

    elif setting.i == 1:
        train_c_mask1 = (train_dataset.real_concepts[:,0] == 1) | (train_dataset.real_concepts[:,0] == 3) | (train_dataset.real_concepts[:,0] == 5) |  \
                        (train_dataset.real_concepts[:,0] == 7) | (train_dataset.real_concepts[:,0] == 9)
        train_c_mask2 = (train_dataset.real_concepts[:,1] == 1) | (train_dataset.real_concepts[:,1] == 3) | (train_dataset.real_concepts[:,1] == 5) |  \
                        (train_dataset.real_concepts[:,1] == 7) | (train_dataset.real_concepts[:,1] == 9)
        train_mask = np.logical_and(train_c_mask1, train_c_mask2)

        test_c_mask1 = (test_dataset.real_concepts[:,0] == 1) | (test_dataset.real_concepts[:,0] == 3) | (test_dataset.real_concepts[:,0] == 5) |  \
                       (test_dataset.real_concepts[:,0] == 7) | (test_dataset.real_concepts[:,0] == 9)
        test_c_mask2 = (test_dataset.real_concepts[:,1] == 1) | (test_dataset.real_concepts[:,1] == 3) | (test_dataset.real_concepts[:,1] == 5) |  \
                       (test_dataset.real_concepts[:,1] == 7) | (test_dataset.real_concepts[:,1] == 9)
        test_mask = np.logical_and(test_c_mask1, test_c_mask2)

    else:
        return NotImplementedError('Impossible choice of task')
    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]
    train_dataset.concepts = train_dataset.concepts[train_mask]
    test_dataset.concepts  = test_dataset.concepts[test_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    print('COncepts')
    print(train_dataset.concepts[:20])
    print('Classes')
    print(train_dataset.targets[:20])

    train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True)# num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=setting.args.batch_size, shuffle=False)# num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader