import torch.nn
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input,start_dim=1)


class UnFlatten(nn.Module):
    def forward(self, input, hidden_channels, dim):
        return input.reshape(input.size(0), hidden_channels, dim[0], dim[1])


class PairCNNEncoder(nn.Module):
    NAME = 'PairCNNEncoder'
    def __init__(self, img_channels=1, hidden_channels=32, latent_dim=8, label_dim = 20, dropout=0.5,img_concept_size=28):
        super(PairCNNEncoder, self).__init__()

        self.img_concept_size=img_concept_size
        self.channels=3
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = int(latent_dim / 2)
        self.label_dim = label_dim
        self.unflatten_dim = (3, 7)

        self.backbone1 =torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.img_channels,
                out_channels=self.hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1),  # 2*hidden_channels x 7 x 14
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=self.hidden_channels * 2,
                out_channels=self.hidden_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1),  # 4*hidden_channels x 2 x 7
            torch.nn.ReLU(),
            Flatten(),
            nn.Linear(
            in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]*(3/7)) ,
            out_features=self.latent_dim)
        )


    def forward(self, x):
        # MNISTPairsEncoder block 1
        x1 = x[:,:,:,:self.img_concept_size]
        x2 = x[:,:,:,self.img_concept_size:]
        mu1= self.backbone1(x1)
        mu2= self.backbone1(x2)
        return mu1, mu2

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


class CNNClassifier(nn.Module):
    def __init__(self, img_channels=1, hidden_channels=32, latent_dim=8, dropout=0.5):
        super(CNNClassifier, self).__init__()

        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.unflatten_dim = (3, 7)

        self.backbone =torch.nn.Sequential(
                            nn.Conv2d(
                                in_channels=self.img_channels,
                                out_channels=self.hidden_channels,
                                kernel_size=4,
                                stride=2,
                                padding=1
                            ),
                            torch.nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Conv2d(
                                in_channels=self.hidden_channels,
                                out_channels=self.hidden_channels * 2,
                                kernel_size=4,
                                stride=2,
                                padding=1),  # 2*hidden_channels x 7 x 14
                            torch.nn.ReLU(),
                            nn.Dropout(p=dropout),
                            nn.Conv2d(
                                in_channels=self.hidden_channels * 2,
                                out_channels=self.hidden_channels * 4,
                                kernel_size=4,
                                stride=2,
                                padding=1),  # 4*hidden_channels x 2 x 7
                            torch.nn.ReLU(),
                            Flatten(),
                            nn.Linear(
                            in_features=int(4 * self.hidden_channels * self.unflatten_dim[0] * self.unflatten_dim[1]) ,
                            out_features=self.latent_dim)
                        )
    def forward(self, x):
        # MNISTPairsEncoder block 1
        x = x.view(-1,1,28,56)
        output= self.backbone(x)
        # mu = torch.cat((mu1, mu2), dim = -1)
        return output

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
