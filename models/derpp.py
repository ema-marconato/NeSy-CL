from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.losses import kl_divergence


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default=1,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Penalty weight.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, concepts, not_aug_inputs):
        self.opt.zero_grad()
        outputs, reprs = self.net(inputs)
        loss, losses = self.loss(outputs, labels, reprs, concepts)
        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs, _ = self.net(buf_inputs)
            buf_loss = self.args.alpha * kl_divergence(buf_outputs, buf_logits, dim=1, version=self.args.version)
            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs, _ = self.net(buf_inputs)
            buf_loss += self.args.beta * self.loss(buf_outputs, buf_labels, None, None)[0]
            loss += buf_loss
            losses.append(buf_loss.item())
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs,labels=labels, logits=outputs.data)
        return loss.item(), losses
