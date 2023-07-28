import numpy as np
import torch
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.utils.clevr import LOAD_CLEVR, OOD_CLEVR
from datasets.utils.continual_dataset import ContinualDataset
from backbone.CLEVR_problog import CLEVR_DeepProblog
from typing import Tuple


class CLE4EVR(ContinualDataset):

    NAME = 'cle4vr'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK =4
    N_CLASSES = 4
    N_TASKS = 5
    TRANSFORM = None
    SEQUENCE_LEN=6
    n_digits=10
    batch_size=30
    data_dir="data/clevr_rccn_pretrained_1"


    def get_data_loaders(self, full=False):
        train_dataset,val_dataset,test_dataset=LOAD_CLEVR(self.data_dir, self.i, full,c_sup=self.args.c_sup)
        train_loader = DataLoader(train_dataset,
                                batch_size=self.args.batch_size, shuffle=True)# num_workers=4)
        if not full:
            if self.args.val_search:
                test_loader = DataLoader(val_dataset,
                                   batch_size=self.args.batch_size, shuffle=False)# num_workers=4)
            else:
                test_loader = DataLoader(test_dataset,
                                    batch_size=self.args.batch_size, shuffle=False)# num_workers=4)
            self.i+=1
            self.test_loaders.append(test_loader)
            self.train_loader = train_loader

            return train_loader,test_loader
        else:
            val_loader = DataLoader(val_dataset,
                                   batch_size=self.args.batch_size, shuffle=False)# num_workers=4
            test_loader = DataLoader(test_dataset,
                                    batch_size=self.args.batch_size, shuffle=False)# num_workers=4)
            return train_loader, val_loader, test_loader

    def get_ood_loader(self):
        testset=OOD_CLEVR('data/clevr_ood_test_rcnn_pretrained_1')
        loader = DataLoader(testset, batch_size=self.args.batch_size, shuffle=True)# num_workers=4)
        return loader

    @staticmethod
    def get_backbone(version):
        if version == 'nesy':
            model=CLEVR_DeepProblog()
            return model
        else:
            return None

    @staticmethod
    def get_transform():
        return None #  transforms.ToTensor()

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

    ## BINARY CLASSIFICATION, CLASS-INCREMENTAL
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i + 1,
        np.array(train_dataset.targets) < setting.i + 1 + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i + 1,
        np.array(test_dataset.targets) < setting.i +1 + setting.N_CLASSES_PER_TASK)

    train_dataset.data = np.array( train_dataset.data)[train_mask]
    test_dataset.data = np.array(test_dataset.data)[test_mask]

    train_dataset.scenes = np.array( train_dataset.scenes)[train_mask]
    test_dataset.scenes = np.array( test_dataset.scenes)[test_mask]

    train_dataset.concepts = np.array(train_dataset.concepts)[train_mask]
    test_dataset.concepts  = np.array(test_dataset.concepts)[test_mask]


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

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader,test_loader