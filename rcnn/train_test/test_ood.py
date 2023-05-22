from wandb.apis.public import Api
import wandb
import torch
import pandas as pd
from backbone.CLEVR_problog import CLEVR_DeepProblog
import glob,os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import numpy as np,joblib,glob
from utils.metrics import custom_accuracy_concepts
from models.utils.continual_model import ContinualModel
from typing import Tuple
from tqdm import tqdm


def evaluate(model: ContinualModel, loader, NAME='MNIST', version='nesy') -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model on given loader`.
    :param model: the model to be evaluated
    :param loader: val/test loader
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    model.eval()
    accs, accs_mask_classes = [], []
    c_accs, c_accs_mask_classes = [], []

    c_distr = torch.zeros(10 if NAME != 'cle4vr' else 20)
    card = torch.zeros(10 if NAME != 'cle4vr' else 20)
    total, correct, c_correct = 0, 0, 0
    loss, l_loss, c_loss = 0, 0, 0
    for data in loader:
        with torch.no_grad():
            inputs, labels, concepts, not_aug_inputs= data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            concepts = concepts.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
            outputs, reprs = model(not_aug_inputs) # forward pass



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

        if  NAME == 'cle4vr': total = 336
        accs.append(correct / total * 100)

        c_accs.append(c_correct / total * 100)

    return accs, c_accs


class OOD_CLEVR(torch.utils.data.Dataset):
    def __init__(self, base_path):

        self.base_path = base_path

        self.list_images= glob.glob(os.path.join(self.base_path,"image","*"))
        self.task_number = [0] * len(self.list_images)
        self.img_number = [i for i in range(len(self.list_images))]
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.concept_mask=np.array([False for i in range(len(self.list_images))])
        self.metas=[]
        self.targets=torch.LongTensor([])
        for item in range(len(self.list_images)):
            target_id=os.path.join(self.base_path,"meta",str(self.img_number[item])+".joblib")
            meta=joblib.load(target_id)
            self.metas.append(meta)

    @property
    def images_folder(self):
        return os.path.join(self.base_path,"image")

    @property
    def scenes_path(self):
        return os.path.join(self.base_path,"image")

    def __getitem__(self, item):
        meta=self.metas[item]
        label=meta["target"]
        concepts= meta["concepts"]
        mask= self.concept_mask[item]
        if mask:
            concepts=-torch.ones_like(concepts)
        task_id, img_id = self.task_number[item], self.img_number[item]
        image_id=os.path.join(self.base_path,"image",str(img_id)+".jpg")
        image = pil_loader(image_id)
        return self.transform(image),label,concepts,self.transform(image)

    def __len__(self):
        return len(self.list_images)


api=Api()

dataset= OOD_CLEVR("PATH")
loader=torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=318,
        num_workers=1,
        drop_last=True,
    )



runs = api.runs("NAME", filters={"tags": {"$in": ["paper"]}})

accuracy=[]
c_accuracy=[]
names=[]
superc=[]
print(runs)
count=0
for run in tqdm(runs):
    try:
        id,name,c_sup,version=run.id,run.name,run.config["c_sup"],run.config["version"]
        if version=="nesy":
            model=CLEVR_DeepProblog()
        try:
            model.load_state_dict(torch.load(wandb.restore('model.pth',run_path="NAME"+id,replace=True).name))
            model.eval()
            model=model.cuda()
            acc,c_acc=evaluate(model,loader,NAME="cle4vr",version=version)
            names.append(name)
            accuracy.append(acc[0])
            c_accuracy.append(c_acc[0])
            superc.append(c_sup)
            print(name,acc[0])

        except:
            print("error",id,name,c_sup)
    except:
        print("errore")


df=pd.DataFrame({"name":names,"concept_supervision":superc,"accuracy":accuracy,"concept_accuracy":c_accuracy})
df.to_csv("clevr_results_ood.csv")
print("finish")

