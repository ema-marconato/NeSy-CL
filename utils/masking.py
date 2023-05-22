import torch
from datasets.utils.continual_dataset import ContinualDataset

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    if dataset.NAME == 'pair-seq-mnist':
        outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
        outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
                dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')
    elif dataset.NAME == 'concepts-seq-mnist':
        outputs[:, 0: 2*k * dataset.N_CONCEPTS_PER_TASK] = -float('inf')
        outputs[:, (2*k + 1) * dataset.N_CONCEPTS_PER_TASK + 1:] = -float('inf')
        # print('Classes:', outputs[0])

    elif dataset.NAME == 'shortcut-mnist':
        if k == 0:
            outputs[:, [0,1,2,3,4,5,7,8,9,11,13,14,15,16,17,18]] = -float('inf')
        elif k == 1:
            outputs[:, [0,1,3,5,7,9,11,13,15,17]] = -float('inf')

    elif dataset.NAME == 'cle4vr':
        outputs = outputs


def mask_concepts(reprs: torch.Tensor, dataset: ContinualDataset,
                  k: int, dtype='one-hot' ) -> None:
    '''nesy_continual
    k = 4, c = {0,1,2,3,4,5,6,7,8,9}, l = {8,9}
    k = 5, c = {1,2,3,4,5,6,7,8,9}, l = {10,11}
    k = 6, c = {3,4,5,6,7,8,9}, l = {12,13}
    k = 7, c = {5,6,7,8,9}, l = {14, 15}
    k = 8, c = {7,8,9}, l = {16, 17}
    k = 9, c = {9}, l = {18}
    '''
    if dataset.NAME == 'concepts-seq-mnist':
        ## EXTRACT CONCEPT INCREMENTAL
        if dtype == 'one-hot':
            reprs = reprs.view(-1,20)
            c_1, c_2 = reprs[:, :10], reprs[:, 10:]
            c_1[:, 0: k * dataset.N_CONCEPTS_PER_TASK] = -float('inf')
            c_2[:, 0: k * dataset.N_CONCEPTS_PER_TASK] = -float('inf')
            c_1[:, (k + 1) * dataset.N_CONCEPTS_PER_TASK: dataset.N_TASKS * dataset.N_CONCEPTS_PER_TASK] = -float('inf')
            c_2[:, (k + 1) * dataset.N_CONCEPTS_PER_TASK: dataset.N_TASKS * dataset.N_CONCEPTS_PER_TASK] = -float('inf')

            return torch.stack((c_1, c_2), dim=1)

        elif dtype == 'continous':
            return reprs

    elif dataset.NAME == 'pair-seq-mnist' or dataset.NAME == 'pair-hard-mnist':
        ## EXTRACT CONCEPTS OF SUMS
        if dtype == 'one-hot':
            reprs = reprs.view(-1,20)
            c_1, c_2 = reprs[:, :10], reprs[:, 10:]
            if k < 4:
                c_1[:, 2*(k + 1):] = -float('inf')
                c_2[:, 2*(k + 1):] = -float('inf')
            if k >= 5:
                c_1[:, :2*(k -5) + 1] = -float('inf')
                c_2[:, :2*(k -5) + 1] = -float('inf')
            return torch.stack((c_1, c_2), dim=1)

        elif dtype == 'continous':
            return reprs

    elif dataset.NAME == 'shortcut-mnist':
        reprs = reprs.view(-1,20)
        if k == 0:
            reprs[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]] = -float('inf')
        elif k == 1:
            reprs[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]] = -float('inf')
        else:
            NotImplementedError('Wrong choice of task')
        return reprs

    elif dataset.NAME == 'cle4vr':
        reprs = reprs.view(-1,40)
        reprs[:, 0:2*k] = -float('inf')
        reprs[:, 2*k+2:10+2*k] = -float('inf')
        reprs[:, 10+2*k+2:20+2*k] = -float('inf')
        reprs[:, 20+2*k+2:30+2*k] = -float('inf')
        reprs[:, 30+2*k+2:] = -float('inf')
        return reprs
    else:
        NotImplementedError('wrong choice')


