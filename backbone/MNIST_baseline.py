import torch
from torch import nn
from backbone.utils.encoders import  CNNClassifier

class MNIST_baseline(nn.Module):
    def __init__(self, is_train=True, device='cpu'):

        super(MNIST_baseline, self).__init__()
        self.latent_dim = 64
        self.encoder = CNNClassifier(hidden_channels=64, latent_dim=self.latent_dim )
        self.classifier = torch.nn.Linear(self.latent_dim, 19)
        self.is_train = is_train
        self.device = device
        self.nr_classes = 19

    def forward(self, x):
        x = self.encoder(x)
        reprs = torch.nn.ReLU()(x)
        sums = self.classifier(reprs)
        return sums, reprs