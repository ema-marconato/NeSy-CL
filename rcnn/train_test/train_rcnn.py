import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from datasets.utils.clevr import CLEVR_SIMPLE
import wandb
import torchvision.transforms as T
import os


def test_boundingbox(model,dataloader):
    model.eval()
    batch=next(iter(dataloader))
    imgs,_=batch
    predictions=model(imgs)
    img=imgs[0]
    boxes=predictions[0]["boxes"].cpu().detach().numpy()
    scores= predictions[0]["scores"].cpu().detach().numpy()
    plot_bbox_predicted(img,boxes,scores)
    model.train()
    #


def train(model,dataloader,optim):
    model.train()
    for _ in range(100):
        for idx,data in enumerate(dataloader):
            optim.zero_grad()
            imgs,targets=data
            results=model(imgs,targets)
            loss= results["loss_box_reg"]+results["loss_rpn_box_reg"]
            wandb.log({"Loss":loss.item()})
            loss.backward()
            optim.step()

            if idx %1000==0:
                test_boundingbox(model,dataloader)
                torch.save(model.state_dict(),os.path.join(wandb.run.dir, "model.pt"))
                wandb.save(os.path.join(wandb.run.dir, "model.pt"))
    wandb.finish()

def plot_bbox_predicted(img,boxes,scores):
    all_boxes = []
    # plot each bounding box for this image
    for box, score in zip(boxes,scores):
        # get coordinates and labels
        box_data = {"position" : {
        "minX" : int(box[0].item()),
        "minY" : int(box[1].item()),
        "maxX" : int(box[2].item()),
        "maxY" : int(box[3].item())},
        "class_id" : int(score*100) ,
        "scores" :{ "score" : int(score*100) },
        # optionally caption each box with its class and score
        "domain" : "pixel"}
        all_boxes.append(box_data)
    immagine=wandb.Image(img,boxes = {"predictions": {"box_data": all_boxes}})
    wandb.log({"examples": immagine})

def plot_bbox(img,v_boxes):
    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        # get coordinates and labels
        box_data = {"position" : {
        "minX" : box[0].item(),
        "minY" : box[1].item(),
        "maxX" : box[2].item(),
        "maxY" : box[3].item()},
        "class_id" : 0,
        "scores" :{ "score" : 0 },
        # optionally caption each box with its class and score
        "domain" : "pixel"}
        all_boxes.append(box_data)
    immagine=wandb.Image(img,boxes = {"predictions": {"box_data": all_boxes}})
    wandb.log({"examples": immagine})


def collate_fn(batch):

    return tuple(zip(*batch))

def main():

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True,)
    num_classes = 15 # 14 Classes + 1 background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model= model.cuda()
    optim= torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=0.001, momentum=0.9, weight_decay=0.0005)

    dataset= CLEVR_SIMPLE("PATH","val")

    dataloader= torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=16,
            num_workers=0,
            drop_last=True,
            collate_fn=collate_fn
        )
    wandb.init(project="training rcnn", entity="NAME")
    model.load_state_dict(torch.load("PATH"))
    train(model,dataloader,optim)


import submitit
executor = submitit.AutoExecutor(folder="./rcnn", slurm_max_num_timeout=30)
executor.update_parameters(
        mem_gb=4,
        gpus_per_task=1,
        tasks_per_node=1,  # one task per GPU
        cpus_per_gpu=1,
        nodes=1,
        timeout_min=200,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition="dev",
        slurm_signal_delay_s=120,
        slurm_array_parallelism=1)

executor.submit(main)