import torch
from utils.status import progress_bar
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
from utils.losses import get_loss
from models import get_model
from utils.metrics import custom_accuracy_concepts
from utils.conf import create_path
import wandb
from utils.posthoc_analysis import  get_confusions
from utils.wandb_logger import *
from utils.masking import *
from sklearn.metrics import accuracy_score

def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, task_n=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    c_accs, c_accs_mask_classes = [], []

    c_distr = torch.zeros(20 if dataset.NAME == 'cle4vr' else 10)
    card = torch.zeros(20 if dataset.NAME == 'cle4vr' else 10)
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        if dataset.NAME == 'cle4vr':
            total=500

        c_correct, c_correct_mask_classes = 0.0, 0.0
        if last and k < len(dataset.test_loaders) - 1:
            continue
        for data in test_loader:
            with torch.no_grad():

                inputs, labels, concepts, _ = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                concepts = concepts.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs, reprs  = model(inputs, k) # forward pass
                else:
                    outputs, reprs = model(inputs) # forward pass
                if dataset.NAME != 'cle4vr':
                    total += labels.shape[0]

                ## EVALUATION OF ACCURACY
                _, pred  = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()

                ## CONCEPTS ACCURACY
                if dataset.args.version in ['nesy', 'cbm']:
                    perc_correct, distr, cc = custom_accuracy_concepts(concepts, reprs, dataset.NAME)
                else:
                    perc_correct, distr, cc = 0, 0, 0

                c_correct += perc_correct
                c_distr += distr
                card += cc

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

                    if dataset.args.version in ['nesy', 'cbm']:
                        # if dataset.NAME == "cle4vr":
                        #     c_correct_mask_classes+=(torch.topk(reprs,k=5,sorted=False)[1].cpu().numpy()[:,:4]==concepts.cpu().numpy()).sum()/4
                        # else:
                        a_reprs = mask_concepts(reprs, dataset, k, 'one-hot')
                        c_correct_mask_classes += custom_accuracy_concepts(concepts, a_reprs, dataset.NAME)[0]

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

        c_accs.append(c_correct / total * 100
                     if 'class-il' in model.COMPATIBILITY else 0)
        c_accs_mask_classes.append(c_correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes, c_accs, c_accs_mask_classes, (c_distr / (card+1))

## PERCHE EVALUATION NON LA FA SUL TEST? LA FA EVALUATE

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    print('### Selected device:', model.device)

    model.net.to(model.device)
    results, results_mask_classes = [], []
    c_results, c_results_mask_classes = [], []
    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, args.version, args.c_sup)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)
    if args.wandb is not None:
        print('\n---wandb on\n')
        wandb.init(project=args.project, entity="NAME",name=str(args.model)+"_"+str(args.version))
        wandb.config.update(args)


    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
       print("pretask",t+1)
       model.net.train()
       _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
       random_results_class, random_results_task,c_accs, c_accs_mask_classes, _ = evaluate(model, dataset_copy, task_n=t)

    if dataset_copy.NAME == 'cle4vr':
        dataset_copy.get_ood_loader()

    # print(file=sys.stderr)
    complete_distr = torch.zeros((dataset.N_TASKS, 20 if dataset.NAME == 'cle4vr' else 10))
    for t in range(dataset.N_TASKS):
        model.task=t
        # recreate model for Restart strategy
        if model.NAME == 'restart':
            print()
            print('New model at task', t+1)
            del model
            backbone = dataset.get_backbone(args.version)
            loss     = get_loss(args)
            model    = get_model(args, backbone, loss, dataset.get_transform())
            PATH = f'data/{args.dataset}-{args.version}-start.pt'
            model.net.load_state_dict(torch.load(PATH))
            model.net.to(model.device)

        model.net.train()

        train_loader, test_loader = dataset.get_data_loaders()

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            results[t-1] = results[t-1] + accs[0]
            c_results[t-1] = c_results[t-1] + accs[2]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
                c_results_mask_classes[t-1] = c_results_mask_classes[t-1] + accs[3]

        scheduler = dataset.get_scheduler(model, args)

        print(model.opt)

        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
               continue
            for i, data in enumerate(train_loader): # THIS IS A CONTINUAL DATASET INSTANCE
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, concepts, not_aug_inputs, logits = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    concepts = concepts.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss, losses = model.observe(inputs, labels, concepts, not_aug_inputs, logits) # FORWARD IS HERE
                else:
                    inputs, labels, concepts, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    concepts = concepts.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss, losses = model.observe(inputs, labels, concepts, not_aug_inputs) # FORWARD IS HERE

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i, name='loss')
                    tb_logger.log_loss(losses[0], args, epoch, t, i, name='label-loss')
                    tb_logger.log_loss(losses[1], args, epoch, t, i, name='concept-loss')

                # WANDB LOG
                if args.wandb is not None:
                    if scheduler is not None:
                        wandb.log({"lr":float(scheduler.get_last_lr()[0])})
                    wandb_log_step(model, loss, losses, epoch, t, i)

            if scheduler is not None:
                print()
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        # save-model
        PATH = f'data/{args.version}/{args.model}-exp_{t}.pt'
        create_path(f'data/{args.version}')
        torch.save(model.net.state_dict(), PATH)

        # evaluate accs with mammoth
        accs = evaluate(model, dataset, task_n=t)

        results.append(accs[0])
        results_mask_classes.append(accs[1])
        c_results.append(accs[2])
        c_results_mask_classes.append(accs[3])

        # calculate accuracy on the tasks
        mean_acc = np.mean(accs[:2], axis=1)
        mean_c_acc = np.mean(accs[2:4], axis=1)
        print_mean_accuracy(mean_acc, mean_c_acc, accs, t + 1, dataset.SETTING)

        # additional metrics
        complete_distr[t] = accs[4]

         # obtain confusions
        cf_labels, cf_preds, cf_concepts, cf_z_pred, cf_z = get_confusions(model, dataset)


        # save accuracy on logger
        if args.csv_log:
            csv_logger.log(mean_acc, mean_c_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.asarray(accs), mean_acc, args, t)
            tb_logger.log_c_accuracy(np.asarray(c_accs), mean_c_acc, args, t)
            tb_logger.log_concepts(accs[4], args, t)

        # EVALUATE ACC ON ALL TESTSET
        acc_labels   = accuracy_score(cf_labels, cf_preds)
        
        if cf_concepts.shape[-1] > 1:
            c_pred = np.concatenate(np.split(cf_z_pred, indices_or_sections=cf_concepts.shape[-1], axis=-1), axis=0)
            c_true = cf_concepts.reshape(-1)
        else:
            c_pred = cf_z_pred
            c_true = cf_concepts

        acc_concepts = accuracy_score(c_true, c_pred) if dataset.args.version != 'baseline' else 0



        # save things in wandb
        if args.wandb is not None:
            wandb_log_task(model, task=t,
            accs=accs, # align=align, # log_lh=log_lh, #relevant metrics
            complete_distr=complete_distr, # dis_h=dis_h,  #log_lh_c=log_lh_c, #metrics per concept
            cf=[cf_labels, cf_preds, cf_concepts, cf_z_pred], # confusion matrices
            acc_labels=acc_labels, acc_concepts=acc_concepts, # scores on test-set
            )

    if args.wandb is not None:
        PATH = os.path.join(wandb.run.dir, 'model.pth')
        torch.save(model.net.state_dict(), PATH)
        wandb.save('model.pth')



    # add additional metrics when terminating training
    if args.csv_log and args.wandb:
        bwt, bwt_mask_classes = csv_logger.add_bwt(results, results_mask_classes)
        forget, forget_mask_classes = csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            fwt, fwt_mask_classes = csv_logger.add_fwt(results, random_results_class,
                                    results_mask_classes, random_results_task)
        wand_log_end(bwt, bwt_mask_classes, forget, forget_mask_classes, fwt, fwt_mask_classes, t)

    if args.tensorboard:
        tb_logger.close()
    if args.wandb is not None:
        wandb.finish()
    if args.csv_log:
        csv_logger.write(vars(args))

