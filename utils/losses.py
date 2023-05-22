import torch
from torch.nn.modules import Module
from torch.nn import functional as F


def get_loss(args):
    w_label = args.l_weight
    w_concepts = args.c_weight
    if args.version == 'nesy':
        return NesyConceptLabelLoss(w_label, w_concepts)
    elif args.version == 'cbm':
        return NeuralConceptLabelLoss(w_label, w_concepts)
    else:
        return ClassificationLoss(w_label)

class ClassificationLoss(Module):
    def __init__(self, query_w=1):
        super(ClassificationLoss, self).__init__()
        self.query_w = query_w
        self.target_loss = torch.nn.CrossEntropyLoss()

    def forward(self, query, targets, concepts_pred, real_concepts):
        loss = self.query_w * self.target_loss(query, targets.to(torch.long))
        label_loss = loss.item()
        return loss, [label_loss, 0]

class NesyConceptLabelLoss(Module):
    def __init__(self, query_w=1, c_w=1):
        """loss used for nesy on concepts

        Args:
            query_w (int, optional): it regulates how the problog prediction influences the loss. Defaults to 1.
            c_w (int, optional): it regulates how important are the concepts meaning in the loss. Defaults to 1.
        """
        super(NesyConceptLabelLoss, self).__init__()
        self.query_w = query_w
        self.c_w = c_w
        self.target_loss = F.nll_loss

    def forward(self, query, targets, concepts_pred, real_concepts, flag_sup=None):
        """loss computation.
        it contains 3 terms: 1)problog prediction;2)c1 prediction;3)c2 prediction

        Args:
            query (torch.Tensor): problog output
            targets (torch.Tensor): ground truth to correct the problog prediction
            concepts_pred (torch.Tensor): concepts predictions before problog
            real_concepts (_type_): ground truth to correct the concept prediction
            flag_sup (_type_, optional):useless

        Returns:
            _type_: loss
        """
        # EVALUATE LOSS ON LABELS (positive examples)
        loss = self.query_w * self.target_loss(query.log(), targets.to(torch.long))
        label_loss = loss.item()

        if concepts_pred == None or real_concepts == None:
            return loss, [loss.item(), 0]
        if len(concepts_pred.shape)>2:
            concepts_pred=concepts_pred.view(concepts_pred.shape[0],-1)
        tot_concepts=real_concepts.shape[1]
        concept_size=concepts_pred.size(1) // tot_concepts

        concepts = torch.split(concepts_pred, concept_size, dim= 1)

        for idx,concept in enumerate(concepts):
            concept = concept.view(-1,concept_size)
            real_concept=real_concepts[:,idx]
            mask = ~(real_concept == -1)
            if mask.sum() > 0:
                loss += self.c_w * self.target_loss(concept[mask].log(), real_concept[mask].long())

        concept_loss = loss.item() - label_loss
        return loss, [label_loss,concept_loss]


class NeuralConceptLabelLoss(Module):
    def __init__(self, query_w=1, c_w=1):
        super(NeuralConceptLabelLoss, self).__init__()
        self.query_w = query_w
        self.c_w = c_w
        self.target_loss=torch.nn.CrossEntropyLoss()
        self.concept_loss=torch.nn.CrossEntropyLoss()

    def forward(self, y_pred, targets, concepts_pred, real_concepts):
        # EVALUATE LOSS ON LABELS
        loss = self.query_w * self.target_loss(y_pred, targets.long())
        label_loss = loss.item()

        if real_concepts == None or concepts_pred == None:
            return loss, [loss.item(), 0]

        concepts_predictions = torch.split(concepts_pred, 10, dim= 1)

        for idx,c in enumerate(concepts_predictions):
            c1 = c.view(-1,10)
            rc1 = real_concepts[:,idx]
            mask1 = ~(rc1 == -1)
            # EVALUATE LOSS ON CONCEPTS
            if mask1.sum() > 0:
                loss += self.c_w * self.concept_loss(c1[mask1], rc1[mask1].long())

        concept_loss = loss.item() - label_loss

        return loss, [label_loss, concept_loss]

def l2(input, target):
    z1, z2 = torch.split(input, input.size(1) // 2, dim=1)
    z1 = z1.view(-1,10)
    z2 = z2.view(-1,10)

    c1, c2 = torch.split(target, target.size(1) // 2, dim=1)
    c1 = c1.view(-1,10)
    c2 = c2.view(-1,10)


    loss  = torch.nn.MSELoss(reduction='mean')(z1, c1)
    loss += torch.nn.MSELoss(reduction='mean')(z2, c2)
    return loss

def kl_divergence(input,target, dim=2, version='nesy'):
    T=1
    kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    if dim == 2:
        if (len(input.shape)>2):
            input=input.view(input.shape[0],-1)
            target=target.view(target.shape[0],-1)
        students = torch.split(input, 10, dim=1)
        teachers = torch.split(target, 10, dim=1)
        loss=0
        if version == 'nesy':
            for student,teacher in zip(students,teachers):
                student=(student + 1e-5)/(1+1e-4)
                student=student.log()
                teacher=(teacher + 1e-5)/(1+1e-4)
                teacher=teacher ** (1/T) / torch.sum(teacher ** (1/T),dim=1).view(-1,1)
                teacher=teacher.log()
                loss+=kl(student,teacher)
        else:
            for student,teacher in zip(students,teachers):
                student=F.log_softmax(student,dim=1)
                teacher=F.log_softmax(teacher,dim=1)
                loss+=kl(student,teacher)

        return loss
    elif dim ==1 :
        if version == 'nesy':
            input += 1e-5
            target += 1e-5
            input /= (1 + 1e-5 * input.size(-1))
            target /= (1 + 1e-5 * input.size(-1))
            input = input.log()
            target = target.log()
            loss = kl(input, target)
        else:
            input = torch.log_softmax(input, dim=-1)
            target = torch.log_softmax(target, dim=-1)
            loss = kl(input, target)
        return loss
    else:
        return None
