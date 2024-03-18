import torch
from .sohot_helpers import soft_activation


class Node:

    def __init__(self, smooth_step_param, split_test=None):
        super().__init__()
        # pointer to left (internal) node
        self.left = None
        # pointer to right (internal) node
        self.right = None
        # pointer to left leaf node (if it exists)
        self.left_leaf = None
        # pointer to right leaf node (if it exists)
        self.right_leaf = None
        self.sample_to_node_prob = 1.
        self.smooth_step_param = smooth_step_param
        self.split_test = split_test
        self.sum_g = 0.
        self.orientation_sequence = ""
        self.alpha = 0.3

    def forward(self, x, w):
        # \alpha * S(<x,w>) + (1-\alpha) [[x_A <= split value]]
        split_criterion_test = int(x[self.split_test.feature] < self.split_test.split_at)
        s_prob = soft_activation(torch.dot(w, x).item(), self.smooth_step_param)
        return self.alpha * s_prob + (1.0 - self.alpha) * split_criterion_test
