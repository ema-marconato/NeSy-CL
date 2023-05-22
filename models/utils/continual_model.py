import torch.nn as nn
from torch.optim import Adam
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss # assign loss
        self.args = args
        self.transform = transform
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()
        self.W = torch.zeros((10,10,19), device=self.device)
        self.t = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, concepts: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass


    def end_task(self, dataset):
        self.t += 1

    def get_reguralization_loss(self, W):
        return 0
