import wandb
import glob,os,joblib
import torchvision
from torchvision import transforms
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import glob
import joblib
from PIL import Image


class CLEVR(torch.utils.data.Dataset):
    def __init__(self, base_path,task,split, full=False):

        self.base_path = base_path
        self.task=str(task)
        self.split=split

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.resize= transforms.Compose(
            [transforms.Resize((28,28))]
        )

        if not full:
            self.list_images= glob.glob(os.path.join(self.base_path,"image_task"+self.task,self.split,"*"))
            self.task_number = [task] * len(self.list_images)
            self.img_number = [i for i in range(len(self.list_images))]
        else:
            self.list_images = []
            self.task_number = []
            self.img_number = []
            for i in range(1,6):
                img_loc = glob.glob(os.path.join(self.base_path,"image_task"+str(i),self.split,"*"))
                self.list_images.extend(img_loc)
                self.task_number.extend([i] * len(img_loc))
                self.img_number.extend([i for i in range(len(img_loc))])

        self.metas=[]
        self.targets=torch.LongTensor([])
        for item in range(len(self.list_images)):
            target_id=os.path.join(self.base_path,"scene_task"+str(self.task_number[item]),self.split,str(self.img_number[item])+".joblib")
            meta=joblib.load(target_id)
            self.metas.append(meta)
            concepts=meta["concepts"]
            if np.array_equal(concepts[0].numpy(), concepts[1].numpy()): # same object
                label=3
            elif np.array_equal(concepts[0,10:19].numpy(), concepts[1,10:19].numpy()): # same color
                label=2
            elif np.array_equal(concepts[0,0:9].numpy(), concepts[1,0:9].numpy()): # same shape
                label=1
            else: # different objects
                label=0
            self.targets=torch.concat([self.targets,torch.LongTensor([label])])

    @property
    def images_folder(self):
        return os.path.join(self.base_path,"image_task"+self.task+"/"+self.split)

    @property
    def scenes_path(self):
        return os.path.join(self.base_path,"image_task"+self.task+"/"+self.split)

    def __getitem__(self, item):
        meta=self.metas[item]
        label=self.targets[item]
        concepts=torch.LongTensor((meta["concepts"].numpy().reshape(40)==1).nonzero()[0])
        concepts[1]-=10
        concepts[2]-=20
        concepts[3]-=30
        concepts= concepts.clone()
        # item=str(item)
        task_id, img_id = self.task_number[item], self.img_number[item]
        image_id=os.path.join(self.base_path,"image_task"+str(task_id),self.split,str(img_id)+".png")
        image = Image.open(image_id).convert('RGB')

        bb=meta["boxes"].int().numpy()

        return self.transform(image),label,concepts,bb

    def __len__(self):
        return len(self.list_images)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def get_crop(img,bb):
		x=torchvision.transforms.functional.crop(img, left=bb[0].long(), top=bb[1].long(), width=(bb[2]-bb[0]).long(), height=(bb[3]-bb[1]).long())
		x=transforms.Resize((28,28))(x)
		return x

def overlap(boxes,targets,idx):
		gt_box=targets[idx]
		iou=bb_intersection_over_union(gt_box[0],boxes[0])
		#print("iou",iou)
		if iou>0:
				return boxes[0],boxes[1]
		else:
				return boxes[1],boxes[0]

#first
model=torch.load(wandb.restore('model.pth', run_path="PATH",replace=True).name)

#all
model.eval()
model.cuda()
dest="PATH"


for i in range(1,6):
    train_dataset=CLEVR("PATH",i,"train")
    test_dataset=CLEVR("PATH",i,"test")
    val_dataset=CLEVR("PATH",i,"val")

    train_loader=torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=64,

            num_workers=0,
            drop_last=True,
        )

    test_loader=torch.utils.data.DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=64,
            num_workers=0,
            drop_last=True,
        )
    val_loader=torch.utils.data.DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=64,
            num_workers=0,
            drop_last=True,
        )
    to_pil= transforms.ToPILImage()
    for data_type,loader in zip(["train","validation","test"],[train_loader,val_loader,test_loader]):
        os.makedirs(os.path.join(dest,"image","task"+str(i),data_type),exist_ok=True)
        os.makedirs(os.path.join(dest,"meta","task"+str(i),data_type),exist_ok=True)
        number=0
        for idx2,data in enumerate(loader):
            imgs, targets,concepts,bounding_boxes = data
            imgs = imgs.cuda()
            targets,concepts=targets.detach(),concepts.detach()
            predictions=model(imgs)

            for idx,img in enumerate(predictions):
                bb= img["boxes"]
                scores= img["scores"]
                size=scores.shape[0]
                image=imgs[idx]
                target=targets[idx].cpu()
                img_concept=concepts[idx]

                if size>=2:
                    indeces= scores.topk(2)[1]
                    box1,box2= overlap(bb[indeces],bounding_boxes,idx)

                    crop1= get_crop(image,box1)
                    crop2= get_crop(image,box2)
                    boxes= torch.concatenate((crop1,crop2),dim=2)
                    new_image= to_pil(boxes)
                    new_image.save(os.path.join(dest,"image","task"+str(i),data_type,str(number)+".jpg"))
                    meta={"concepts":img_concept,"target":target}
                    joblib.dump(meta,os.path.join(dest,os.path.join(dest,"meta","task"+str(i),data_type,str(number)+".joblib")))
                    number+=1

