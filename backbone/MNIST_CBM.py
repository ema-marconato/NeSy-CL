import torch
from torch import nn
from backbone.utils.encoders import PairCNNEncoder

class MNIST_CBM(nn.Module):
    def __init__(self, is_train=True, device='cuda'):

        super(MNIST_CBM, self).__init__()
        self.encoder =  PairCNNEncoder(hidden_channels=64, latent_dim=20,dropout=0.5)  # PairMLPEncoder(28*28, z_dim=10) #
        self.W = torch.nn.parameter.Parameter(torch.rand(size=(10,10,19)))
        self.is_train = is_train
        self.device = device
        self.nr_classes = 19

    def forward(self, x):
        z1, z2 = self.encoder(x)
        concepts = torch.cat((z1,z2),dim=1)

        sums = torch.einsum('bi, ijk, bj -> bk', z1, self.W, z2)

        return sums, concepts

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params