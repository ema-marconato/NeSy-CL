
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.losses import kl_divergence

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default=1,
                        help='Penalty weight.')
    return parser


class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, concepts, notaugmented):
        """ reharsal by kl divergence: DER

        Args:
            inputs (torch.Tensor): input tensor
            labels (torch.Tensor): rule ground truth
            concepts (torch.Tensor): concept ground truth
            notaugmented (torch.Tensor): original input tensor

        Returns:
            tuple: losses
        """
        self.opt.zero_grad()
        #loss on current data
        outputs, reprs = self.net(inputs)
        loss, losses = self.loss(outputs, labels, reprs, concepts) # compute loss, inherited from BaseStrategy
        if not self.buffer.is_empty():
            #get data from the buffer and apply KD
            buf_inputs, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs,_ = self.net(buf_inputs)
            buf_loss = self.args.alpha * kl_divergence(buf_outputs, buf_logits, dim=1, version=self.args.version)
            loss += buf_loss
            losses.append(buf_loss.item())
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=notaugmented, logits=outputs.data)

        return loss.item(), losses
