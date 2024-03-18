# Soft Hoeffding Tree
# Differentiable and transparent predictive base learner

import torch
import torch.nn as nn
from .internal_node import Node
from .leaf_node import LeafNode
from river.tree.split_criterion import InfoGainSplitCriterion, GiniSplitCriterion, HellingerDistanceCriterion
import numpy as np
from .sohot_function import SoHoTFunction
from .sohot_helpers import SplitDecision


class SoftHoeffdingTree(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 max_depth=7,
                 smooth_step_param=1.0,
                 alpha=0.3,
                 split_confidence=1e-6,
                 split_criterion='info_gain',    # 'gini'
                 tie_threshold=0.05,
                 grace_period=600,
                 remove_poor_attrs=False,
                 min_branch_fraction=0.01,
                 seed=None):
        """

        :param input_dim: Input dimension
        :type input_dim: int
        :param output_dim: Output dimension
        :type output_dim: int
        :param max_depth: Maximum depth the tree can reach
        :type max_depth: int
        :param smooth_step_param: Smooth-step parameter 1/gamma from Tree Ensemble Layer
        :type smooth_step_param: float
        :param alpha: Transparency regulator with 0 <= alpha <= 1, values closer to 0 yield to a more transparent SoHoT
        :type alpha: float
        :param split_confidence:
        :type split_confidence: float
        :param split_criterion: Split criterion (Information gain, Gini index or Hellinger Distance)
        :type split_criterion: str
        :param tie_threshold: Threshold below which a split will be forced to break ties
        :type tie_threshold: float
        :param grace_period: Number of instances a leaf should observe between split attempts
        :type grace_period: int
        :param remove_poor_attrs:
        :type remove_poor_attrs: bool
        :param min_branch_fraction: The minimum percentage of observed data required for branches resulting from split candidates
        :type min_branch_fraction: float
        :param seed: Random seed
        :type seed: int
        """
        super(SoftHoeffdingTree, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.smooth_step_param = smooth_step_param
        self.alpha = alpha      # Transparency parameter to control trade-off in gating function
        self.split_confidence = split_confidence

        # River denotes this parameter tau not tie_threshold
        self.tie_threshold = tie_threshold
        self.grace_period = grace_period
        self.remove_poor_attrs = remove_poor_attrs

        if seed is not None: torch.manual_seed(seed)
        # Add 't' (=top) weight in ordered torch dictionary
        self.weights = torch.nn.ParameterDict({'t': nn.Parameter(torch.FloatTensor(output_dim).uniform_(-0.01, 0.01),
                                                                 requires_grad=True)})
        self.root = LeafNode()
        self.root.sample_to_node_prob = 1.

        # Set split criterion
        if split_criterion.__eq__('info_gain'):
            self.split_criterion = InfoGainSplitCriterion(min_branch_fraction=min_branch_fraction)
        elif split_criterion.__eq__('gini'):
            self.split_criterion = GiniSplitCriterion(min_branch_fraction=min_branch_fraction)
        elif split_criterion.__eq__('hellinger'):
            self.split_criterion = HellingerDistanceCriterion(min_branch_fraction=min_branch_fraction)
        else:
            self.split_criterion = InfoGainSplitCriterion()
            print("Invalid split_criterion option {}', will use default '{}'".format(split_criterion, 'info_gain'))

        self.growth_allowed = True

    def total_node_cnt(self):
        return sum(1 for _ in self.parameters())

    def decision_node_cnt(self):
        n = sum(1 for _ in self.parameters())
        return (n - 1) // 2

    @staticmethod
    def weights_to_list(weights):
        return zip(*[(k, v) for k, v in weights.items()])

    def forward(self, x, y=None):
        """
        Forward x through soft Hoeffding tree.
        :param x: Input tensor.
        :type x: Torch.tensor of dimension input_dim
        :param y: True label of the input tensor y.
        :type y: Torch.tensor of dimension output_dim
        :return: Raw probabilities.
        :rtype: Torch.tensor of dimension output_dim
        """
        in_keys, in_list = SoftHoeffdingTree.weights_to_list(self.weights)

        output, _ = SoHoTFunction.apply(x, self.smooth_step_param, self.max_depth, self, y, in_keys, *in_list)
        return output

    def add_parameter(self, name, shape, device, init_tensor=None):
        # self.weights.update({name: torch.FloatTensor(shape).uniform_(-0.01, 0.01).to(device)})
        if init_tensor is not None:
            self.weights.update({name: nn.Parameter(init_tensor, requires_grad=True)})
        else:
            self.weights.update({name: nn.Parameter(torch.FloatTensor(shape).uniform_(-0.01, 0.01), requires_grad=True)})

    def remove_parameter(self, name):
        self.weights.pop(name)

    @staticmethod
    def _hoeffding_bound(range_val, confidence, n):
        return np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))


    def split_leaf(self, leaf, previous_node, split_decision_summary, children_stats):
        device = self.weights[leaf.orientation_sequence].device
        self.remove_parameter(leaf.orientation_sequence)
        new_internal_node, leaf_init_tensor = leaf.split(previous_node, self.output_dim, self.smooth_step_param,
                                       split_decision_summary, children_stats)
        # Set parameter alpha for gating through the tree
        new_internal_node.alpha = self.alpha
        # Set init value to w := (0, ..., 0, 0.01 or 0.1, 0, ..., 0) with entry != 0 on position of split feature
        # init_tensor = torch.zeros(self.input_dim, dtype=torch.float32, device=device)
        # init_tensor[split_decision_summary.feature] = 0.1
        self.add_parameter(new_internal_node.orientation_sequence, self.input_dim, device)
        # Set init values from LeafNode.split function
        self.add_parameter(new_internal_node.left_leaf.orientation_sequence, self.output_dim, device, leaf_init_tensor[0])
        self.add_parameter(new_internal_node.right_leaf.orientation_sequence, self.output_dim, device, leaf_init_tensor[1])
        # self.add_parameter(new_internal_node.left_leaf.orientation_sequence, self.output_dim, device)
        # self.add_parameter(new_internal_node.right_leaf.orientation_sequence, self.output_dim, device)
        if previous_node is None:
            self.root = new_internal_node
            self.root.sample_to_node_prob = 1.

    # Modified version of hoeffding_tree 'attempt_to_spit'
    def attempt_to_split(self, leaf, previous_node):
        #  If the samples seen so far are not from the same class and maximum depth is reached then:
        #  (Regression: If the target's variance is high at the leaf node, then:)
        #         1. Find split candidates and select the top 2.
        #         2. Compute the Hoeffding bound.
        #         3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
        #            3.1 Replace the leaf node by a split node.
        #            3.2 Add a new leaf node on each branch of the new split node.
        #            3.3 Update tree's metrics
        new_internal_node = None
        if not leaf.observed_class_distribution_is_pure():
            best_split_suggestions = leaf.best_split_suggestions(self.split_criterion)
            best_split_suggestions.sort()
            # print("best_split_suggestions", [b.merit for b in best_split_suggestions])
            should_split = False
            if len(best_split_suggestions) < 2:
                # null split is no criterion to split, null is always added
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(self.split_criterion.range_of_merit(leaf.stats),
                                                        self.split_confidence, leaf.total_weight)
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                suggestion_merit_comparison = best_suggestion.merit - second_best_suggestion.merit
                if suggestion_merit_comparison > hoeffding_bound or hoeffding_bound < self.tie_threshold:
                    should_split = True
                # Remove poor attributes (=not promising attributes).
                # See "Mining High-Speed Data Streams" by P. Domingos, G. Hulten.
                if self.remove_poor_attrs:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for suggestion in best_split_suggestions:
                        poor_test = best_suggestion.merit - suggestion.merit > hoeffding_bound
                        if suggestion.feature and poor_test:
                            poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.feature is not None:
                    # Finally, split leaf node
                    # to remember the split criterion for transparency
                    split_decision_summary = SplitDecision(feature=split_decision.feature,
                                                           split_at=split_decision.split_info)
                    self.split_leaf(leaf, previous_node, split_decision_summary, split_decision.children_stats)
        return new_internal_node

    # Traverse the Soft Hoeffding Tree in postorder to do the backward pass from TEL (-> fractional tree)
    def postorder_traversal(self, X):
        if isinstance(self.root, LeafNode):
            return [self.root]
        return self.postorder_traversal_(self.root, X)

    def postorder_traversal_(self, node, X):
        postorder = []
        if node is not None:
            if isinstance(node, Node):
                weight_vec = self.weights[node.orientation_sequence]
                routing_left_prob = node.forward(X, weight_vec)
                if node.left is None and routing_left_prob > 0.:
                    postorder = self.postorder_traversal_(node.left_leaf, X)
                elif node.left is not None and routing_left_prob > 0.:
                    postorder = self.postorder_traversal_(node.left, X)
                if node.right is None and routing_left_prob < 1.:
                    postorder += self.postorder_traversal_(node.right_leaf, X)
                elif node.right is not None and routing_left_prob < 1.:
                    postorder += self.postorder_traversal_(node.right, X)
            postorder.append(node)
        return postorder

    def __str__(self):
        output = ""
        to_traverse = [self.root]
        last_level_internal = []
        while to_traverse:
            i = to_traverse.pop()
            node_weight = self.weights[i.orientation_sequence]
            if isinstance(i, Node):
                current_depth = (len(i.orientation_sequence) - 1)
                output += "    " * current_depth + "{}: Internal weight = {}, \n".format(current_depth,
                                                                                       node_weight.data.numpy())
                if i.split_test:
                    output += "    " * current_depth + "   {}\n".format(i.split_test)
                if i.right_leaf is None:
                    to_traverse.append(i.right)
                else:
                    to_traverse.append(i.right_leaf)
                    last_level_internal.append(i)
                if i.left_leaf is None:
                    to_traverse.append(i.left)
                else:
                    to_traverse.append(i.left_leaf)
                    last_level_internal.append(i)
            # leaf nodes
            else:
                d = i.depth
                output += "    " * d + "{}: Leaf weight = {}, Stats: {}\n".format(d, node_weight.data.numpy(), i.stats)
        return output
