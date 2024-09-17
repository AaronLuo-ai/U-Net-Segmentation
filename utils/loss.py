import torch.nn as nn
import torch
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        print("inputs.shape", inputs.shape)
        print("targets.shape", targets.shape)
        print("inputs.dtype", inputs.dtype)
        print("targets.dtype", targets.dtype)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Calculate intersection
        intersection = (inputs * targets)
        # intersection = [(inputs * targets).nonzero(as_tuple=True)].numel() # Keep it as a tensor, do not call .item()
        print("intersection shape: ", intersection.shape)
        intersection_area = torch.count_nonzero(intersection, dim=0)
        inputs_area = torch.count_nonzero(intersection, dim=0)
        targets_area = torch.count_nonzero(targets, dim=0)
        dice = (2. * intersection_area + smooth) / (inputs_area + targets_area + smooth)
        print("targets_area : ", targets_area)
        print("intersection_area : ", intersection_area)
        print("inputs_area: ", inputs_area)
        # inputs_area = [inputs * targets.nonzero(as_tuple=True)]
        # targets_area = [inputs * targets.nonzero(as_tuple=True)]
        # print("targets: ", targets)
        # print("inputs: ", inputs)
        # Calculate 2 * intersection + smooth
        # intersection_plus_smooth = 2. * intersection + smooth  # Result is still a tensor

        # Calculate cardinality by counting non-zero elements
        # nonzero_elements = tensor[inputs.nonzero(as_tuple=True)]

        # target_cardinality = torch.nonzero(targets).size(0)
        # print("cardinality input: ", input_cardinality)
        # print("cardinality output: ", target_cardinality)
        # Calculate Dice coefficient using cardinality
        # dice = intersection_plus_smooth / (inputs + targets + smooth)
        # print("2. * intersection + smooth = ", 2 * intersection + smooth,
        #       " input_cardinality + target_cardinality + smooth: ", input_cardinality + target_cardinality + smooth)

        return 1 - dice  # Return as tensor for gradient tracking


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        # Apply log_softmax to inputs to get log probabilities
        log_probs = F.log_softmax(inputs, dim=1)  # dim=1 as we're working with class predictions along that axis

        # Create one-hot encoding for the targets (assuming targets are class indices)
        one_hot_targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)

        # Compute the cross-entropy loss as the negative log likelihood
        loss = -torch.sum(one_hot_targets * log_probs) / inputs.size(0)  # Mean over batch size

        return loss
