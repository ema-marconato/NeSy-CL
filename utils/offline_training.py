import torch
from utils.status import progress_bar
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
import wandb
import sys


def evaluate(model: ContinualModel, loader, NAME='MNIST', version='nesy') -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model on given loader`.
    :param model: the model to be evaluated
    :param loader: val/test loader
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, _ = []
    c_accs, _ = []

    c_distr = torch.zeros(10 if model.args.dataset != 'cle4vr' else 20)
    card = torch.zeros(10 if model.args.dataset != 'cle4vr' else 20)
    total, correct, c_correct = 0, 0, 0
    loss, l_loss, c_loss = 0, 0, 0
    for data in loader:
        with torch.no_grad():

            inputs, labels, concepts, not_aug_inputs= data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            concepts = concepts.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
            outputs, reprs = model(not_aug_inputs) # forward pass

            l, ls = model.loss(outputs, labels, reprs, concepts)
            loss += l
            l_loss += ls[0]
            c_loss += ls[1]

            total += labels.shape[0]

            ## EVALUATION OF ACCURACY
            _, pred  = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()

            ## CONCEPTS ACCURACY

            if version in ['nesy', 'cbm']:
                perc_correct, distr, cc = custom_accuracy_concepts(concepts, reprs, name=NAME)
            else:
                perc_correct, distr, cc = 0, 0, 0

            c_correct += perc_correct
            c_distr += distr
            card += cc
        if  NAME == 'cle4vr': total = 500
        accs.append(correct / total * 100)

        c_accs.append(c_correct / total * 100)

    print('\nTotal Loss', loss.item() / len(loader))
    print('Labels Loss', l_loss / len(loader), 'Concepts loss', c_loss / total)


    model.net.train(status)
    return accs, c_accs

def train_offline(model: ContinualModel, dataset: ContinualDataset,
                  args: Namespace) -> None:
    """
    The offline training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    print('### Selected device:', model.device)
    model.net.to(model.device)
    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, args.version, args.c_sup)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    if args.wandb is not None:
        wandb.init(project=args.project, entity="NAME",name=str(args.model)+"_"+str(args.version))
        wandb.config.update(args)
    model.net.train()
    train_loader, val_loader, test_loader = dataset.get_data_loaders(full=True)
    scheduler = dataset.get_scheduler(model, args)
    for epoch in range(model.args.n_epochs):
        for i, data in enumerate(train_loader): # THIS IS A CONTINUAL DATASET INSTANCE

            inputs, labels, concepts, not_aug_inputs = data
            inputs, labels= inputs.to(model.device), labels.to(model.device)
            concepts = concepts.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)

            model.opt.zero_grad()
            outputs, reprs = model.net(not_aug_inputs)

            loss, losses = model.loss(outputs, labels.long(), reprs, concepts)
            loss.backward()
            wandb.log({"loss":loss.item()})
            model.opt.step()
            progress_bar(i, len(train_loader), epoch, 'J', loss.item())

        if args.tensorboard:
            tb_logger.log_loss(loss, args, 0, epoch, i, name='loss')
            tb_logger.log_loss(losses[0], args, epoch, 0,  i, name='label-loss')
            tb_logger.log_loss(losses[1], args, epoch, 0,  i, name='concept-loss')

            if i%100==0:
                print()
                print("Evaluation on Validation")
                accs, c_accs = evaluate(model, val_loader,NAME=dataset.NAME, version=args.version)
                tb_logger.log_loss(np.mean(accs), args,  epoch, 0, i, name='val-accuracy-labels' )
                tb_logger.log_loss(np.mean(c_accs), args, epoch, 0, i, name='val-accuracy-concepts' )

        if scheduler is not None:
            scheduler.step()

        if args.tensorboard:
            accs, c_accs = evaluate(model, test_loader,NAME=dataset.NAME, version=args.version)
            wandb.log({"acc labels":np.mean(accs),"acc concepts":sum(c_accs) / len(c_accs)})

            tb_logger.log_loss(np.mean(accs), args, epoch, 0, i, name='val-accuracy-labels' )
            tb_logger.log_loss(np.mean(c_accs), args, epoch, 0, i, name='val-accuracy-concepts' )
    # quit()
    # obtain confusions
    print()
    print('Evaluation on test-set')
    accs, c_accs = evaluate(model, test_loader,NAME=dataset.NAME, version=args.version)

    # calculate accuracy on the tasks
    mean_acc = np.mean(accs)
    mean_c_acc = np.mean(c_accs)

    print('Accuracy', mean_acc)
    print('Concept Acc', mean_c_acc)


    if args.tensorboard:
        tb_logger.close()
        # wandb.finish()
    if args.csv_log:
        csv_logger.write(vars(args))