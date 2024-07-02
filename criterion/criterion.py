import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class MacroAverageMAELoss(nn.Module):
    def __init__(self, n_classes):
        super(MacroAverageMAELoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, inputs, targets):
        total_mae = 0.0
        
        for class_index in range(self.n_classes):
            class_mask = (targets == class_index)
            class_preds = inputs[class_mask]
            class_targets = targets[class_mask]
            
            if class_targets.numel() > 0:
                class_mae = F.l1_loss(class_preds, class_targets, reduction='mean')
                total_mae += class_mae

        return total_mae / self.n_classes


class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        mse_loss = self.mse(inputs, targets)
        error = torch.abs(inputs - targets)
        focal_weight = (1 + error) ** self.gamma
        focal_mse_loss = torch.mean(focal_weight * mse_loss)
        return focal_mse_loss


class QWKLoss(nn.Module):
    def __init__(self, n_classes=6):
        super(QWKLoss, self).__init__()
        self.n_classes = n_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        W = torch.zeros((n_classes, n_classes), device=self.device)
        for i in range(n_classes):
            for j in range(n_classes):
                W[i, j] = (i - j) ** 2
        self.W = W

    def forward(self, logits, labels):
        logits = torch.round(logits).clamp(0, self.n_classes - 1).long()
        labels = torch.round(labels).clamp(0, self.n_classes - 1).long()

        logits_one_hot = F.one_hot(logits, num_classes=self.n_classes).float()
        labels_one_hot = F.one_hot(labels, num_classes=self.n_classes).float()

        O = torch.matmul(labels_one_hot.transpose(0, 1), logits_one_hot)
        total = O.sum()
        E = torch.matmul(labels_one_hot.sum(dim=0).view(-1, 1), logits_one_hot.sum(dim=0).view(1, -1)) / total

        num = torch.sum(self.W * O)
        den = torch.sum(self.W * E) + 1e-10  
        return 1 - (num / den)


class WeightedDenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()

        return loss


def get_criterion(config):
    valid_criterion_types = ['MSELoss','RMSELoss','MacroAverageMAELoss', 'FocalMSELoss', 'QWKLoss', 'WeightedDenseCrossEntropy']
    assert config.criterion.criterion_type in valid_criterion_types, \
        f"Invalid criterion type. Expected one of {valid_criterion_types} but got {config.criterion.criterion_type}"

    if config.criterion.criterion_type == 'MacroAverageMAELoss':
        return MacroAverageMAELoss(n_classes=config.criterion.n_classes)

    elif config.criterion.criterion_type == 'FocalMSELoss':
        return FocalMSELoss(gamma=config.criterion.gamma)

    elif config.criterion.criterion_type == 'QWKLoss':
        return QWKLoss(n_classes=config.criterion.n_classes)

    elif config.criterion.criterion_type == 'WeightedDenseCrossEntropy':
        return WeightedDenseCrossEntropy()

    elif config.criterion.criterion_type == 'RMSELoss':
        return RMSELoss(
            eps=config.criterion.rmse_loss.eps,
            reduction=config.criterion.rmse_loss.reduction
        )
    elif config.criterion.criterion_type == 'MSELoss':
        return nn.MSELoss()