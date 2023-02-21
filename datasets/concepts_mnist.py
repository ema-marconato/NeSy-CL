import numpy as np
import torch
from torchvision import datasets
import numpy as np
import torch
import torch.nn.functional as F
from datasets.utils.mnist_creation import load_2MNIST
from torch.utils.data import DataLoader
from datasets.utils.continual_dataset import ContinualDataset
from backbone.MNIST_DeepProblog import MNIST_DeepProblog
from backbone.MNIST_CBM import MNIST_CBM
import numpy as np
from typing import Tuple


class ConceptsSequentialMNIST(ContinualDataset):

    NAME = 'concepts-seq-mnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 3
    N_CONCEPTS_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = None
    SEQUENCE_LEN = 2
    n_digits=10
    batch_size=30

    def get_data_loaders(self, full=False):
        train_dataset,val_dataset,test_dataset=load_2MNIST(self.args.batch_size)
        if not full:
            if not self.args.val_search:
                train, test = store_masked_loaders(train_dataset.dataset, test_dataset, self)
                return train, test
            else:
                train, val = store_masked_loaders(train_dataset.dataset, val_dataset, self)
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
        else:
            return None

    @staticmethod
    def get_transform():
        return None

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
    def get_scheduler(model, *args):
        model.opt = torch.optim.Adam(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
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
    ## CONCEPT INCREMENTAL
    train_mask = np.logical_and(np.array(train_dataset.concepts) >= setting.i,
        np.array(train_dataset.concepts) < setting.i + setting.N_CONCEPTS_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.concepts) >= setting.i,
        np.array(test_dataset.concepts) < setting.i + setting.N_CONCEPTS_PER_TASK)

    train_mask = np.logical_and(train_mask[:,0], train_mask[:,1])
    test_mask = np.logical_and(test_mask[:,0], test_mask[:,1])

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.concepts = train_dataset.concepts[train_mask]
    test_dataset.concepts  = test_dataset.concepts[test_mask]

    if setting.args.c_sup < 1:
        # select examples to remove supervision
        mask = np.random.rand(len(train_dataset.concepts)) > setting.args.c_sup
        #flag_unsup = -1 * torch.ones_like((train_dataset.concepts.size()[1:] ))
        train_dataset.concepts[mask] = -1

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    print('Trainset size:', len(train_dataset))
    print('Testset size:',  len(test_dataset))

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True)# num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False)# num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CONCEPTS_PER_TASK
    return train_loader, test_loader