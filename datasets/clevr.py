import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import numpy as np,joblib,glob

def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    '''TODO:adjust'''

    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

def LOAD_CLEVR(data_dir,task, full,c_sup):
    '''TODO:adjust'''
    dataset_train = CLEVR(data_dir,task, "train", full)
    dataset_val = CLEVR(data_dir,task, "val", full)
    dataset_test= CLEVR(data_dir,task, "test", full)

    if c_sup < 1:
        # select examples to remove supervision
        mask = np.random.rand(len(dataset_train.concept_mask)) > c_sup
        dataset_train.concept_mask[mask] = True

    return dataset_train,dataset_val,dataset_test

CLASSES = {
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
}

class CLEVR(torch.utils.data.Dataset):
    N_CLASSES = 20
    def __init__(self, base_path,task,split, full=False):
        '''TODO:adjust'''

        self.base_path = base_path
        self.task=str(task+1)
        self.split=split

        if not full:
            self.list_images= glob.glob(os.path.join(self.base_path,"image","task"+self.task,self.split,"*"))
            self.task_number = [task+1] * len(self.list_images)
            self.img_number = [i for i in range(len(self.list_images))]
        else:
            self.list_images = []
            self.task_number = []
            self.img_number = []
            for i in range(1,6):
                img_loc = glob.glob(os.path.join(self.base_path,"image","task"+str(i),self.split,"*"))
                self.list_images.extend(img_loc)
                self.task_number.extend([i] * len(img_loc))
                self.img_number.extend([i for i in range(len(img_loc))])
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.concept_mask=np.array([False for i in range(len(self.list_images))])
        self.metas=[]
        self.targets=torch.LongTensor([])
        for item in range(len(self.list_images)):
            target_id=os.path.join(self.base_path,"meta","task"+str(self.task_number[item]),self.split,str(self.img_number[item])+".joblib")
            meta=joblib.load(target_id)
            self.metas.append(meta)

    @property
    def images_folder(self):
        return os.path.join(self.base_path,"image_task"+self.task+"/"+self.split)

    @property
    def scenes_path(self):
        return os.path.join(self.base_path,"image_task"+self.task+"/"+self.split)

    def __getitem__(self, item):
        meta=self.metas[item]
        label=meta["target"]
        concepts= meta["concepts"]
        mask= self.concept_mask[item]
        if mask:
            concepts=-torch.ones_like(concepts)
        task_id, img_id = self.task_number[item], self.img_number[item]
        image_id=os.path.join(self.base_path,"image","task"+str(task_id),self.split,str(img_id)+".jpg")
        image = pil_loader(image_id)

        if hasattr(self, 'logits'):
            return self.transform(image),label,concepts,self.transform(image), self.logits[item]
        return self.transform(image),label,concepts,self.transform(image)

    def __len__(self):
        return len(self.list_images)

class OOD_CLEVR(torch.utils.data.Dataset):
    def __init__(self, base_path):

        self.base_path = base_path
        self.list_images= glob.glob(os.path.join(self.base_path,"image","*"))
        self.task_number = [0] * len(self.list_images)
        self.img_number = [i for i in range(len(self.list_images))]
        self.transform = transforms.Compose([transforms.ToTensor()])
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
