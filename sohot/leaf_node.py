from .internal_node import Node
import torch
from river.tree.utils import BranchFactory
from river.tree.splitter import GaussianSplitter, TEBSTSplitter
import river.stats


class LeafNode:

    def __init__(self, initial_stats=None):
        super().__init__()
        self.sample_to_node_prob = 0.
        self.splitter = GaussianSplitter()
        # Each attribute should be observed by a splitter
        self.splitters = {}
        # observes all class values in the current leaf node
        self.stats = initial_stats
        self._disabled_attrs = set()
        # Sum over leaves l in the right or left subtree of a node i, where sum_g = sum over P({x->l})<dL/dT, o_l>
        # (see algorithm 2 from TEL)
        self.sum_g = 0.
        self.orientation_sequence = "t"     # t for top or root
        self.depth = 0
        self.last_split_attempt_at = 0

    @property
    def stats(self):
        return self._stats

    @stats.setter
    def stats(self, stats):
        self._stats = stats if stats is not None else {}

    @property
    def total_weight(self):
        return sum(self.stats.values())

    def update_stats(self, y, weight=1.0):
        try:
            self.stats[y] += weight
        except KeyError:
            self.stats[y] = weight
            self.stats = dict(sorted(self.stats.items()))

    # see river/tree/nodes/leaf
    def update_splitters(self, x, y, weight=1.0):
        for idx, x_i in enumerate(x):
            if idx in self._disabled_attrs:
                continue
            x_i = x_i.item()
            try:
                splitter = self.splitters[idx]
            except KeyError:
                splitter = self.splitter.clone()
                # for every attribute is an observer splitter
                self.splitters[idx] = splitter
            splitter.update(x_i, y, weight)

    # see river/tree/nodes/leaf
    def best_split_suggestions(self, criterion):
        binary_split = True
        best_suggestions = []
        pre_split_dist = self.stats
        # Add null split as an option
        null_split = BranchFactory()
        best_suggestions.append(null_split)
        for att_id, splitter in self.splitters.items():
            best_suggestion = splitter.best_evaluated_split_suggestion(criterion, pre_split_dist, att_id, binary_split)
            best_suggestions.append(best_suggestion)
        return best_suggestions

    def observed_class_distribution_is_pure(self):
        count = 0
        for weight in self.stats.values():
            if weight != 0:
                count += 1
                if count == 2:
                    break
        return count < 2

    @staticmethod
    def forward(prob_reach_leaf, leaf_weight):
        return torch.mul(leaf_weight, prob_reach_leaf)

    def split(self, previous_node, output_dim, smooth_step_param, split_test, children_stats):
        # create new leaf node and one parent node
        new_internal_node = Node(smooth_step_param, split_test)
        is_left = False
        if previous_node is not None:
            is_left = (previous_node.left_leaf == self)
        if previous_node is not None:
            if is_left:
                previous_node.left = new_internal_node
            else:
                previous_node.right = new_internal_node
        left = LeafNode(output_dim)
        right = LeafNode(output_dim)
        # Update resulting statistics for the new leaf nodes
        # Note: Empty the statistics, leaf prediction is based on the weight vectors
        #       Empty stats cause slower growth and the class distribution from stats is not needed for leaf prediction
        left_init_weight, right_init_weight = None, None
        if children_stats:
            desired_len = 0.025
            left_stats_dist = torch.tensor([children_stats[0].get(target_idx, 0) for target_idx in range(output_dim)])
            left_init_weight = torch.mul(torch.div(left_stats_dist, torch.norm(left_stats_dist)), desired_len)
            right_stats_dist = torch.tensor([children_stats[1].get(target_idx, 0) for target_idx in range(output_dim)])
            right_init_weight = torch.mul(torch.div(right_stats_dist, torch.norm(right_stats_dist)), desired_len)
        left.stats, right.stats = {}, {}

        new_internal_node.left_leaf = left
        new_internal_node.right_leaf = right
        new_internal_node.orientation_sequence = self.orientation_sequence
        left.orientation_sequence = self.orientation_sequence + "l"
        left.depth = self.depth + 1
        right.orientation_sequence = new_internal_node.orientation_sequence + "r"
        right.depth = self.depth + 1
        # remove leaf pointer from previous node
        if is_left:
            previous_node.left_leaf = None
        elif not is_left and previous_node is not None:
            previous_node.right_leaf = None
        # remember probability to reach the node
        new_internal_node.sample_to_node_prob = self.sample_to_node_prob
        return new_internal_node, (left_init_weight, right_init_weight)

    def disable_attribute(self, att_id):
        if att_id in self.splitters:
            del self.splitters[att_id]
            self._disabled_attrs.add(att_id)
