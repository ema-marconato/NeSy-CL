import numpy as np
import torch
from sklearn.linear_model import Lasso


def backward_transfer(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = list()
    for i in range(1, n_tasks):
        li.append(results[i-1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)

def custom_accuracy_concepts(concepts, reprs, name):
    if name in ['pair-seq-mnist', 'shortcut-mnist']:
        reprs = reprs.view(-1,20)
    zs = torch.split(reprs, 10,   dim=1)
    cs = torch.split(concepts, 1, dim=1)
    correct = 0
    for c, z in zip(cs, zs):
        _, z_pred = torch.max(z, 1)
        correct += torch.sum(z_pred==c.view(-1)).item() / len(cs)

    if name in ['pair-seq-mnist', 'shortcut-mnist']:
        scores = torch.zeros(10)
        card = torch.zeros(10)
        z_pred = torch.cat([torch.argmax(zs[0].view(-1,10), dim=-1).view(-1,1),
                            torch.argmax(zs[1].view(-1,10), dim=-1).view(-1,1)], dim=0)
        concepts = torch.cat([cs[0], cs[1]], dim=0)
    elif name == 'cle4vr':
        scores = torch.zeros(20)
        card = torch.zeros(20)
        z_pred_obj_1 = torch.cat([torch.argmax(zs[0].view(-1,10), dim=-1).view(-1,1),
                                    torch.argmax(zs[1].view(-1,10), dim=-1).view(-1,1)], dim=1)
        c_obj_1 = torch.cat([cs[0].view(-1,1), cs[1].view(-1,1)], dim=1)

        z_pred_obj_2 = torch.cat([torch.argmax(zs[2].view(-1,10), dim=-1).view(-1,1),
                                    torch.argmax(zs[3].view(-1,10), dim=-1).view(-1,1)], dim=1)
        c_obj_2 = torch.cat([cs[2].view(-1,1), cs[3].view(-1,1)], dim=1)

        z_pred = torch.cat([z_pred_obj_1, z_pred_obj_2], dim=0)
        concepts = torch.cat([c_obj_1, c_obj_2], dim=0)
    else:
        return NotImplementedError('Dataset not implemented.')

    # evaluate scores on each concepts
    for l in range( reprs.size(1) // 20 ):
        for i in range(10):
            mask = (concepts[:,l] == i % 10).view(-1)
            card[i+10*l] += mask.sum().item()
            scores[i+10*l] += (z_pred[:,l][mask]==concepts[:,l][mask]).sum().item()
    return correct, scores, card

def correctness(x,y):
    '''
    Define whether a continuous number is correct w.r.t. the real integer
    '''
    h = torch.abs(x - y)
    return (h < 0.5).sum().item()

def alignment_score(z: np.array, concepts: np.array, version):

    assert len(concepts) == len(z)
    # assert concepts.shape[1] == 1, concepts.shape
    if version not in ['nesy', 'cbm']:
        return 0, [0]*10

    # one-hot codification of ground-truth concepts

    c_all = np.split(concepts, 2, axis=1)

    del concepts
    for k, cs in enumerate(c_all):
        c = np.zeros((cs.size, cs.max() + 1))
        c[np.arange(cs.size), cs] = 1
        if k == 0:
            concepts = c
        else:
            concepts = np.concatenate([concepts, c], axis=1)

    model = Lasso(alpha=0.01, max_iter=10 ** 4)
    model.fit(z, concepts)

    W = np.abs(model.coef_) + 1e-6 # small offset 1e-6

    ## W is K x L, where K = dim C and L = dim Z
    K, L = W.shape

    A = np.zeros_like(W)
    rho = np.zeros(L) # coeffs of shannon
    h = np.zeros(L)   # shannon entropies

    for i in range(L):
        A[:, i] = (W[:,i]) / np.sum(W[:,i])
        rho[i] = np.sum(W[:,i]) / np.sum(W)
        h[i] = - np.sum( A[:,i] * np.log(A[:,i]) ) / np.log(K) # log K is maximum value of entropy

    align = 1 - np.sum( rho * h )

    return align, 1 - h

def log_likelihood(z: np.array, c: np.array, version):
    if version not in ['nesy', 'cbm']:
        return 0, [0]*10

    log_l_c = np.zeros(10)
    for i in np.unique(c):
        mask_c = (c==i)
        log_l_c[i] =   np.mean(np.log( z[mask_c][:, i])) / np.log(10)

    return np.mean(log_l_c[~(log_l_c == 0)]  ), log_l_c
