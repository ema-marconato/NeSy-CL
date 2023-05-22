from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Naive Strategy.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Restart(ContinualModel):
    NAME = 'restart'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Restart, self).__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, concepts, not_aug_inputs):
        self.opt.zero_grad()
        outputs, reprs = self.net(inputs)
        loss, losses = self.loss(outputs, labels, reprs, concepts)
        loss.backward()
        self.opt.step()
        return loss.item(), losses
