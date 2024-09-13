import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # To prevent division by zero

    def forward(self, prediction, target):
        # Convert prediction and target to binary masks using a threshold of 0.5
        prediction_binary = (prediction > 0.5).float()  # Convert to binary (0 or 1)
        target_binary = (target > 0.5).float()  # Convert to binary (0 or 1)

        # Flatten the tensors to calculate DSC
        prediction_flat = prediction_binary.view(-1)
        target_flat = target_binary.view(-1)

        # Calculate the intersection and union for Dice score
        intersection = (prediction_flat * target_flat).sum()  # Intersection of prediction and ground truth
        total = prediction_flat.sum() + target_flat.sum()  # Union

        # Calculate Dice Coefficient
        dice_score = (2. * intersection + self.smooth) / (total + self.smooth)

        # Return 1 - dice_score as loss (since we want to minimize the loss)
        return 1 - dice_score