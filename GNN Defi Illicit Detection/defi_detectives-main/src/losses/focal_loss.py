import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return: tensor containing the weights for each class
    """

    n_classes = len(cls_num_list)
    # Claude helped to debug warning for converting to tensor
    if (isinstance(cls_num_list, np.ndarray)):
        cls_tensor = torch.from_numpy(cls_num_list).float()
    else:
        cls_tensor = cls_num_list.detach().clone().requires_grad_(False)
    e_n = (1 - beta**cls_tensor) / (1 - beta)
    alpha = 1 / e_n
    per_cls_weights = alpha * (n_classes / sum(alpha))

    return per_cls_weights

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0, beta=0.9999, device='cuda'):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.device = device
        self.beta = beta
        # Claude recommended to only reweight once
        self.weight = weight
        if self.weight is not None:
            device = self.device if torch.cuda.is_available() else 'cpu'
            self.weight = reweight(weight, self.beta).to(device)

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None

        # Ensure weight is on the same device as input
        if self.weight is not None and self.weight.device != input.device:
            self.weight = self.weight.to(input.device)

        p = F.softmax(input, dim=-1) #softmax of the logits
        # Claude recommended to prevent loss explosion
        p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
        p_t = p[torch.arange(p.shape[0], device=input.device), target]
        w_t = self.weight[target.long()]
        focal_loss = ((1 - p_t)**self.gamma) * torch.log(p_t)
        class_loss = -w_t * focal_loss
        loss = torch.mean(class_loss)

        return loss