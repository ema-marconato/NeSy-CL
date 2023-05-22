import wandb
import numpy as np
import torch


def wandb_log_step(model, loss, losses, epoch, t, i):
    wandb.log({"loss":loss,
               "label-loss": losses[0],
               "concept-loss": losses[1],
                "epoch": epoch,
                "task":t,
                "step": i})

    if hasattr(model, 'buffer') and len(losses) >= 3:
        wandb.log({'buf-loss': losses[2], "epoch": epoch,"task":t,"step": i})
    else:
        wandb.log({'buf-loss': 0, "epoch": epoch,"task":t,"step": i})

def wandb_log_task(model, **kwargs):
    # log accuracies
    t = kwargs['task']
    accs = kwargs['accs']
    mean_acc = np.mean(accs[:4], axis=1)
    wandb.log({"class-il":mean_acc[0],"task-il":mean_acc[1],
                "c-class-il":mean_acc[2],"c-task-il":mean_acc[3],
                'task-label-acc': accs[0][t], 'task-concept-acc': accs[2][t],
                "task": t})

    # log specific to each concept
    complete_distr = kwargs['complete_distr']
    L = complete_distr.shape[1]
    for i in range(L):
        wandb.log({'acc_c=%i'%i: complete_distr[t,i], "task":t})

    # log confusion matrices
    cf_labels, cf_preds, cf_concepts, cf_z_pred = kwargs['cf']
    K = cf_labels.max()
    wandb.log({'confusion-preds': wandb.plot.confusion_matrix(None, cf_labels, cf_preds,  class_names=[str(i) for i in range(K+1)]),
                'task': t})

    if len(cf_concepts.shape) == 1:
        M = cf_concepts.max()
        wandb.log({'confusion-concepts': wandb.plot.confusion_matrix(None, cf_concepts, cf_z_pred, class_names=[str(i) for i in range(M+1)])})
    else:
        mask = (cf_concepts[:,0] == 2*t) | (cf_concepts[:,0] == 2*t + 1)
        mask = mask & ((cf_concepts[:,1] == 2*t) | (cf_concepts[:,1] == 2*t + 1))
        l = len(mask) // 2

        mask = (mask[:l]) & (mask[l:])

        wandb.log({'confusion-preds-task': wandb.plot.confusion_matrix(None, cf_labels[mask], cf_preds[mask],
                    class_names=[str(i) for i in range(K+1)]), 'task': t})
        for i in range(cf_concepts.shape[1]):
            M = cf_concepts[:, i].max()
            wandb.log({'confusion-concepts-%i'%i: wandb.plot.confusion_matrix(None, cf_concepts[:, i], cf_z_pred[:, i],
                        class_names=[str(i) for i in range(M+1)])})

    # log perfs on test-set
    wandb.log({"overall-acc-labels": kwargs['acc_labels']})
    wandb.log({"overall-acc-concepts": kwargs['acc_concepts']})


def wand_log_end(bwt, bwt_mask_classes, forget, forget_mask_classes, fwt, fwt_mask_classes, t):
    # log score metrics
    wandb.log({'bwt': bwt, 'bwt_mask_classes': bwt_mask_classes,
                'forget': forget, 'forget_mask_classes': forget_mask_classes,
                'fwt': fwt, 'fwt_mask_classes': fwt_mask_classes,
                "task":t})
