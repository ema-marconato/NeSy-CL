from utils.conf import base_path
import os
from argparse import Namespace
import numpy as np


class TensorboardLogger:
    def __init__(self, args: Namespace, setting: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.settings = [setting]
        if setting == 'class-il':
            self.settings.append('task-il')
        self.loggers = {}
        self.name = args.model
        for a_setting in self.settings:
            self.loggers[a_setting] = SummaryWriter(
                os.path.join(base_path(), 'tensorboard_runs', a_setting, self.name))
        config_text = ', '.join(
            ["%s=%s" % (name, getattr(args, name)) for name in args.__dir__()
             if not name.startswith('_')])
        for a_logger in self.loggers.values():
            a_logger.add_text('config', config_text)


    def get_name(self) -> str:
        """
        :return: the name of the model
        """
        return self.name

    def log_accuracy(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     args: Namespace, task_number: int) -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """
        mean_acc_common, mean_acc_task_il = all_mean_accs
        for setting, a_logger in self.loggers.items():
            mean_acc = mean_acc_task_il\
                if setting == 'task-il' else mean_acc_common
            index = 1 if setting == 'task-il' else 0
            accs = [all_accs[index][kk] for kk in range(len(all_accs[0]))]
            for kk, acc in enumerate(accs):
                a_logger.add_scalar('acc_task%02d' % (kk + 1), acc,
                                    task_number * args.n_epochs)
            a_logger.add_scalar('acc_mean', mean_acc, task_number * args.n_epochs)

    def log_c_accuracy(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     args: Namespace, task_number: int) -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """
        mean_acc_common, mean_acc_task_il = all_mean_accs
        for setting, a_logger in self.loggers.items():
            mean_acc = mean_acc_task_il if setting == 'task-il' else mean_acc_common
            accs = [all_accs[kk] for kk in range(len(all_accs))]
            for kk, acc in enumerate(accs):
                a_logger.add_scalar('c_acc_task%02d' % (kk + 1), acc,
                                    task_number * args.n_epochs)
            a_logger.add_scalar('c_acc_mean', mean_acc, task_number * args.n_epochs)


    def log_concepts(self, c_distr,
                     args: Namespace, task_number: int) -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """
        for setting, a_logger in self.loggers.items():
            for i in range(len(c_distr)):
                a_logger.add_scalar('acc_c=1', c_distr[i], task_number * args.n_epochs)


    def log_loss(self, loss: float, args: Namespace, epoch: int,
                 task_number: int, iteration: int, name='label-loss') -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar(name, loss, task_number * args.n_epochs + epoch)

    def log_loss_gcl(self, loss: float, iteration: int) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param iteration: the current iteration
        """
        for a_logger in self.loggers.values():
            a_logger.add_scalar('loss', loss, iteration)

    def log_confusions(self, cf_preds, cf_reprs, task):
        '''
        Add confusion matrices to tensorboard.
        '''
        for a_logger in self.loggers.values():
            a_logger.add_figure("Labels confusion matrix", cf_preds, task)
            a_logger.add_figure("Concepts confusion matrix", cf_reprs, task)


    def close(self) -> None:
        """
        At the end of the execution, closes the logger.
        """
        for a_logger in self.loggers.values():
            a_logger.close()
