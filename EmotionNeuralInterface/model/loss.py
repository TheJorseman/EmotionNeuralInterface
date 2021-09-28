import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.distances import LpDistance

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=self.margin)
        self.eps = 1e-9

    """
    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

    """

    def forward(self, output1, output2, target, size_average=True):
        #return self.loss(output1, output2, target)
        distances = torch.sqrt((output2 - output1).pow(2).sum(1)) # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps)))
        return losses.mean() if size_average else losses.sum()


class ContrastiveLossPML(nn.Module):
    def __init__(self, use_pairwise=True):
        super(ContrastiveLossPML, self).__init__()
        self.pos, self.neg, self.distance = self.get_pos_neg_vals(use_pairwise)
        self.loss = losses.ContrastiveLoss(pos_margin=self.pos, neg_margin=self.neg, distance=self.distance)

    def get_pos_neg_vals(self, use_pairwise):
        output = (0,1, LpDistance(power=2))
        if not use_pairwise:
            return (1,0, CosineSimilarity())
        return output

    def forward(self, z1, z2, y):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        #z = F.cosine_similarity(z1, z2)
        z = self.distance(z1, z2)
        return self.loss(z, y.squeeze())


class CosineEmbeddingLoss(nn.Module):
    def __init__(self, margin):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.loss = nn.CosineEmbeddingLoss(margin=margin)
    
    def forward(self, z1, z2, y):
        z1 = F.normalize(z1.squeeze(), p=2, dim=1)
        z2 = F.normalize(z2.squeeze(), p=2, dim=1)
        return self.loss(z1, z2, y.squeeze())


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, ):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._cosine_simililarity
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")


    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)