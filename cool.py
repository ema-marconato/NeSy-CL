
from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.losses import kl_divergence
import torch

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default=1,help='Penalty weight.')
    parser.add_argument('--beta', type=float, default=1,help='Penalty weight.')
    return parser


class COOL(ContinualModel):
    NAME = 'cool'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(COOL, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0


    def end_task(self, dataset):
        self.task += 1

    def observe(self, inputs, labels, concepts, notaugmented):
        # self.time += 1
        self.opt.zero_grad()

        outputs, reprs = self.net(inputs)
        loss, losses = self.loss(outputs, labels, reprs, concepts)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_concepts, buf_logits, buf_task_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs, buf_reprs = self.net(buf_inputs)

            buf_loss = self.args.alpha * kl_divergence(buf_reprs, buf_logits, version=self.args.version) #c_cross_entropy(buf_reprs, buf_logits) # F.mse_loss(buf_reprs, buf_logits)
            loss += buf_loss

            losses.append(buf_loss.item())
            loss += self.loss(buf_outputs, buf_labels, None, None)[0]  # label reharsal


        loss.backward()
        self.opt.step()
        # if self.time > self.wait_time:
        self.buffer.add_data(examples=notaugmented,
                            labels=labels,
                            concepts=concepts,
                            logits=reprs.data,
                            task_labels=self.task * torch.ones(len(notaugmented)) )

        return loss.item(), losses
