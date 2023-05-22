import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser

class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, concepts, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        outputs, reprs = self.net(inputs)
        loss, losses = self.loss(outputs, labels, reprs[:real_batch_size], concepts)
        losses.append(loss.item())
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs,labels=labels[:real_batch_size])
        return loss.item(), losses
