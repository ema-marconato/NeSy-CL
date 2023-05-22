import torch
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from utils.losses import  kl_divergence

def average_entropy(concepts, task_labels, N_TASKS):

    entropies = torch.zeros(N_TASKS, device=concepts.device)
    for i in task_labels.unique():
        mask_c = (task_labels == i)
        p = concepts[mask_c]
        entropy_task = - torch.sum(p * p.log()) / mask_c.sum()
        entropies[i] = entropy_task
    return torch.sum(entropies) / (i+1), entropies

def analyze_buffer(model: ContinualModel, dataset: ContinualDataset, args, task):
    """
    Analyze loss on buffer during tasks

    Args:
        model (_type_): _description_
    """
    mem = model.buffer.get_all_data()

    distr, c_distr = torch.zeros((19,)), torch.zeros((10,))
    buf_labels, buf_concepts = None, None
    if model.NAME in ['dcr', 'cool']:
        buf_inputs, buf_labels, buf_concepts, buf_logits, buf_task_labels = mem

        # EVALUATE DISTR ON BUFFER
        if args.c_sup == 1:
            for i in range(10):
                mask_c0 = (buf_concepts[:,0] == i)
                mask_c1 = (buf_concepts[:,1] == i)

                c_distr[i] = mask_c0.sum() + mask_c1.sum()
            for i in range(19):
                mask_l = (buf_labels == i)
                distr[i] = mask_l.sum()

            torch.save(distr, f'data/buffer/labels-distr-{model.NAME}-{args.version}-{args.buffer_size}-task_{task}')
            torch.save(c_distr, f'data/buffer/concepts-distr-{model.NAME}-{args.version}-{args.buffer_size}-task_{task}')

        # evaluate confusion on buffer

        z1, z2 = torch.split(buf_logits, buf_logits.size(1) // 2, 1)
        all_concepts = torch.cat([z1,z2])
        all_tasks = torch.cat([buf_task_labels, buf_task_labels])

        avg_h, h = average_entropy(all_concepts, all_tasks, dataset.N_TASKS)

    elif model.NAME == 'der':
        buf_inputs, buf_logits = mem
    elif model.NAME == 'derpp':
        buf_inputs,  buf_labels, buf_logits = mem

    elif model.NAME in ['gdumb','ceres']:
        buf_inputs, buf_labels, buf_concepts = mem
    elif model.NAME == 'gem':
        buf_inputs, buf_labels, _ = mem
    else:
        buf_inputs, buf_labels = mem

    with torch.no_grad():
        buf_inputs = buf_inputs.to(device=model.device)
        buf_outputs, buf_reprs = model(buf_inputs)

        loss = torch.zeros(size=(), device=model.device)
        if model.NAME in ['dcr', 'cool']:
            loss += kl_divergence(buf_reprs, buf_logits)
            if model.NAME == 'dcr_taskpp':
                loss += model.loss(buf_outputs, buf_labels, None, None)[0]
        elif model.NAME in ['der', 'derpp']:
            loss += kl_divergence(buf_logits, buf_outputs, dim=1)
            if model.NAME == 'derpp':
                loss += model.loss(buf_outputs, buf_labels, None, None)[0]
        else:
            loss += model.loss(buf_outputs, buf_labels, None, None)[0]
    if model.NAME in ['dcr', 'cool']:
        return loss.item(), avg_h, h
    else:
        return loss.item(), None, None