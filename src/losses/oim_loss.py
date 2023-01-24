# Global imports
import torch
import torch.nn.functional as F
from torch import nn


# Refactored OIM loss with safe float16 computation
class OIMLossSafe(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, use_cq=True):
        super(OIMLossSafe, self).__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids

        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0
            self.cq = None

    def forward(self, inputs, label):
        # Normalize inputs
        inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        outputs_labeled = inputs.mm(self.lut.t().clone())
        outputs_labeled[:, bad_lut_mask] = -1
        outputs_labeled[bad_pos_idx, bad_label] = 1

        # Compute cosine similarity of inputs with CQ
        if self.cq is not None:
            outputs_unlabeled = inputs.mm(self.cq.t().clone())
            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # Multiply projections by (inverse) temperature scalar
        projected *= self.oim_scalar

        # Compute loss
        ## for numerical stability with float16, we divide before computing the sum to compute the mean
        ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce
        _loss_oim = F.cross_entropy(projected, label, ignore_index=self.ignore_index, reduction='none')
        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()

        # Compute LUT and CQ updates
        with torch.no_grad():
            targets = label
            for x, y in zip(inputs, targets):
                if y < len(self.lut):
                    self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x, dim=0)
                elif self.cq is not None:
                    self.cq[self.header_cq] = x
                    self.header_cq = (self.header_cq + 1) % self.cq.size(0)

        # Return loss
        return loss_oim
