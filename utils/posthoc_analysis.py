import numpy as np
import torch
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset

def expand_cf(cf_matrix, dim=10):
    cf = np.zeros((dim,dim))
    cf[:cf_matrix.shape[0], :cf_matrix.shape[1]] = cf_matrix
    return cf

def get_confusions(model: ContinualModel, dataset: ContinualDataset, get_ood=False):
    if not get_ood:
        _, _, test_loader = dataset.get_data_loaders(full=True)
    else:
        test_loader = dataset.get_ood_loader()

    images, labels, concepts = None, None, None
    for image, label, concept, _   in test_loader:
        if images is None:
            images = image
            labels = label
            concepts = concept
        else:
            images = torch.cat([images, image])
            labels = torch.cat([labels, label])
            concepts = torch.cat([concepts, concept])

    preds, reprs = model(images.to(device=model.device))

    if dataset.args.version in ['nesy', 'cbm']:
        # concepts: 1 dim for MNIST, 2 dim for CLEVR
        #  z: 10 dim for MNIST, 20 dim for CLEVR
        if (len(reprs.shape)>2):
                reprs=reprs.view(reprs.shape[0],-1)
        zs = torch.split(reprs, 10,   dim=1)
        cs = torch.split(concepts, 1, dim=1)

        if dataset.NAME in ['pair-seq-mnist', 'shortcut-mnist']:

            z = torch.cat([zs[0].view(-1,10), zs[1].view(-1,10)])
            z_pred = torch.cat([torch.argmax(zs[0].view(-1,10), dim=-1).view(-1,1),
                                torch.argmax(zs[1].view(-1,10), dim=-1).view(-1,1)], dim=0)
            concepts = torch.cat([cs[0], cs[1]], dim=0)
        elif dataset.NAME == 'cle4vr':
            z_obj_1 = torch.cat([zs[0].view(-1,10), zs[1].view(-1,10)], dim=1)
            z_pred_obj_1 = torch.cat([torch.argmax(zs[0].view(-1,10), dim=-1).view(-1,1),
                                      torch.argmax(zs[1].view(-1,10), dim=-1).view(-1,1)], dim=1)
            c_obj_1 = torch.cat([cs[0].view(-1,1), cs[1].view(-1,1)], dim=1)

            z_obj_2 = torch.cat([zs[2].view(-1,10), zs[3].view(-1,10)], dim=1)
            z_pred_obj_2 = torch.cat([torch.argmax(zs[2].view(-1,10), dim=-1).view(-1,1),
                                      torch.argmax(zs[3].view(-1,10), dim=-1).view(-1,1)], dim=1)
            c_obj_2 = torch.cat([cs[2].view(-1,1), cs[3].view(-1,1)], dim=1)

            z = torch.cat([z_obj_1, z_obj_2], dim=0)
            z_pred = torch.cat([z_pred_obj_1, z_pred_obj_2], dim=0)
            concepts = torch.cat([c_obj_1, c_obj_2], dim=0)
        else:
            return NotImplementedError('ot implemented dataset')

        z = z.detach().cpu().numpy()
        z_pred = z_pred.detach().cpu().numpy()
        concepts = concepts.numpy()
    else:
        z = None
        z_pred = None
        concepts = None

    # pass everything to np
    preds = torch.argmax(preds, dim=-1).detach().cpu().numpy()
    labels = labels.numpy()

    return labels, preds, concepts, z_pred, z
