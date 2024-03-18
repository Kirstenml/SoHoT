import torch.nn as nn
import torch
from .sohot import SoftHoeffdingTree


class SoftHoeffdingTreeLayer(nn.Module):
    def __init__(self, input_dim, output_dim, ssp=1.0, max_depth=7, trees_num=10, split_confidence=1e-6,
                 tie_threshold=0.05, grace_period=600, is_target_class=True, average_output=False,
                 seeds=None, alpha=0.3):
        super(SoftHoeffdingTreeLayer, self).__init__()
        if seeds is None: seeds = [None]*trees_num
        else: torch.manual_seed(42)
        self.bn = nn.BatchNorm1d(input_dim, momentum=0.99, eps=0.001)       # BN adds randomness
        self.sohots = nn.ModuleList([SoftHoeffdingTree(input_dim, output_dim, max_depth=max_depth,
                                                       smooth_step_param=ssp, alpha=alpha,
                                                       split_confidence=split_confidence,
                                                       tie_threshold=tie_threshold, grace_period=grace_period,
                                                       seed=seeds[t])
                                     for t in range(trees_num)])
        self.is_target_class = is_target_class
        self.average_output = average_output

    def forward(self, x, y=None):
        x = self.bn(x)
        outputs = [None]*len(self.sohots)
        for i, shtree in enumerate(self.sohots):
            outputs[i] = shtree(x, y)
        if self.average_output:
            x = torch.stack(outputs, dim=0).mean(dim=0)
        else:
            x = torch.stack(outputs, dim=0).sum(dim=0)
        return x
