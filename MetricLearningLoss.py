import torch
import torch.nn as nn

class MetricLearningLoss(nn.Module):
    def __init__(self, sigma=0.2, omega=1.0):
        super(MetricLearningLoss, self).__init__()
        self.sigma = sigma
        self.omega = omega

    def __call__(self, outputs, labels, sigma=0.2, omega=1):
        # Calculate pairwise distances squared
        # pairwise_distances_sq = torch.cdist(features, features, p=2)**2
        # print("pairwise_distances_sq", pairwise_distances_sq)
        #
        # # Create a mask for pairs with the same class and pairs with different classes
        # same_class_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # print("same_class_mask", same_class_mask)
        # diff_class_mask = 1.0 - same_class_mask
        # print("diff_class_mask", diff_class_mask)
        #
        # # Calculate the loss for pairs with the same class
        # loss_same_class = -(pairwise_distances_sq * same_class_mask) / (2 * self.sigma**2)
        # loss_same_class += 0.5 * torch.log(pairwise_distances_sq * same_class_mask / (2 * self.sigma**2))
        # loss_same_class = torch.sum(loss_same_class) / (torch.sum(same_class_mask) - len(labels))
        # print("loss_same_class", loss_same_class)
        #
        # # Calculate the loss for pairs with different classes
        # loss_diff_class = (pairwise_distances_sq * diff_class_mask) / (2 * self.omega**2)
        # loss_diff_class -= 0.5 * torch.log(pairwise_distances_sq * diff_class_mask / (2 * self.omega**2))
        # loss_diff_class = torch.sum(loss_diff_class) / (torch.sum(diff_class_mask) - len(labels))
        # print("loss_diff_class", loss_diff_class)
        #
        # # Total loss
        # total_loss = loss_same_class + loss_diff_class

        # Compute pairwise Euclidean distances between points in the output space
        pairwise_distances = torch.cdist(outputs, outputs, p=2)

        # Get the square of the distances
        pairwise_distances_squared = pairwise_distances.pow(2)

        # Construct a mask to identify same-class and different-class pairs
        same_class_mask = labels.view(-1, 1) == labels.view(1, -1)
        different_class_mask = ~same_class_mask

        # Calculate the contribution of same-class pairs to the loss
        same_class_distances_squared = pairwise_distances_squared[same_class_mask]
        same_class_loss = -0.5 * (same_class_distances_squared / (2 * sigma ** 2)).sum()

        # Calculate the contribution of different-class pairs to the loss
        different_class_distances_squared = pairwise_distances_squared[different_class_mask]
        different_class_loss = 0.5 * (different_class_distances_squared / (2 * omega ** 2)).sum()

        # Combine the same-class and different-class contributions to the loss
        total_loss = same_class_loss + different_class_loss

        return total_loss