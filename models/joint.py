from torch.optim import SGD, Adam
import wandb
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.old_concepts = []
        self.current_task = 0
        self.version = args.version
        self.criterion= nn.CrossEntropyLoss()

    def end_task(self, dataset):
        if dataset.SETTING != 'domain-il':
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets.astype(int)))
            self.old_concepts.append(torch.tensor(dataset.train_loader.dataset.concepts.astype(int)))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS: return

            # reinit network
            self.net = dataset.get_backbone(self.version)
            self.net.to(self.device)
            self.net.train()
            self.opt = Adam(self.net.parameters(), lr=self.args.lr)
            scheduler = dataset.get_scheduler(self)

            # prepare dataloader
            all_data, all_labels, all_concepts = None, None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                    all_concepts = self.old_concepts[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])
                    all_concepts = np.concatenate([all_concepts, self.old_concepts[i]])

            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            temp_dataset = ValidationDataset(all_data, all_labels, all_concepts, transform=transform)
            loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels, concepts = batch
                    inputs, labels = inputs.to(dtype=torch.float, device=self.device), labels.to(self.device)
                    concepts = concepts.to(self.device)

                    self.opt.zero_grad()
                    outputs, reprs = self.net(inputs)
                    loss, _ = self.loss(outputs, labels.long(), reprs, concepts)
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())

                if scheduler is not None:
                    scheduler.step()

    def observe(self, inputs,new_image, target_set, img_class_ids, image_id, img_expl, table_expls): # inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        out_pred,_ = self.net(inputs)
        # convert sorting gt target set and gt table explanations to match the order of the predicted table
        loss= torch.nn.CrossEntropyLoss()(out_pred,img_class_ids)
        loss.backward()
        self.opt.step()
        return loss.item(),[loss.item(),loss.item()]

