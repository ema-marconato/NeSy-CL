import torch
from datasets import get_dataset
from torch.optim import SGD
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Learning without Forgetting.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Penalty weight.')
    parser.add_argument('--softmax_temp', type=float, default=2,
                        help='Temperature of the softmax function.')
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class Lwf(ContinualModel):
    NAME = 'lwf'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Lwf, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.dataset = get_dataset(args)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        nc = get_dataset(args).N_CLASSES ## here 1 is for MNIST
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)

    def begin_task(self, dataset):
        self.net.eval()
        if self.current_task > 0:
            if not self.args.version == 'nesy':
                self.cbm=True
                opt = SGD([{'params':self.net.W ,"lr":self.args.lr}])

                for _ in range(self.args.n_epochs):
                    for i, data in enumerate(dataset.train_loader):
                        inputs, labels, concepts, not_aug_inputs = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        concepts = concepts.to(self.device)
                        opt.zero_grad()
                        with torch.no_grad():
                            _, reprs = self.net(inputs)
                            z1,z2= torch.split(reprs,10,dim=1)
                        mask = self.eye[(self.current_task + 1) * self.cpt - 1] ^ self.eye[self.current_task * self.cpt - 1]

                        outputs =torch.einsum('bi, ijk, bj -> bk', z1, self.net.W, z2)[:, mask]
                        loss, losses = self.loss(outputs, labels - self.current_task * self.cpt, reprs, concepts)
                        loss.backward()
                        opt.step()
            else:
                self.cbm=False

            logits = []
            with torch.no_grad():
                for i in range(0, dataset.train_loader.dataset.data.shape[0], self.args.batch_size):
                    inputs = torch.stack([torch.as_tensor(dataset.train_loader.dataset.__getitem__(j)[3])
                                          for j in range(i, min(i + self.args.batch_size,len(dataset.train_loader.dataset)))])
                    log = self.net(inputs.to(dtype=torch.float, device=self.device))[0].cpu()
                    logits.append(log)
            setattr(dataset.train_loader.dataset, 'logits', torch.cat(logits))
        self.net.train()

        self.current_task += 1

    def observe(self, inputs, labels, concepts, not_aug_inputs, logits=None):
        self.opt.zero_grad()
        outputs, reprs = self.net(inputs)

        if self.current_task == 9:
            mask = self.eye[-1]
        else:
            mask = self.eye[self.current_task * self.cpt - 1]

        loss, losses = self.loss(outputs[:, mask], labels, reprs, concepts)
        if logits is not None:
            mask = self.eye[(self.current_task - 1) * self.cpt - 1]
            if self.cbm:
                loss += self.args.alpha *  torch.nn.KLDivLoss(log_target=True)(self.logsoft(outputs[:, mask]),self.logsoft(logits[:, mask]))

            else:
                loss += self.args.alpha * modified_kl_div(smooth(logits[:, mask].to(self.device),2, 1),
                                                          smooth(outputs[:, mask], 2, 1))
        loss.backward()
        self.opt.step()
        return loss.item(), losses
