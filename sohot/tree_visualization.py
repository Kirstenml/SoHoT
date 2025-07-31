import networkx as nx
from .internal_node import Node
from .leaf_node import LeafNode
import matplotlib.pyplot as plt
import random
from pathlib import Path
from torch.nn import Softmax
import re

'''
Graph Visualization of Soft Hoeffding Trees
Using the bibliography NetworkX
Note: Edge probabilities are visualized with 3 decimals, therefore probability 1 is not always 1.
    Color: #8FAADC
'''


class SohotVisualization:

    def __init__(self, sohot, schema):
        self.schema = schema
        self.sohot = sohot
        self.c_deepest_path = 1

        self.attribute_list_names = re.findall(r'@attribute\s+(\S+)', str(self.schema.get_moa_header()))
        self.attribute_list = []  # Repeat attribute name in list (numValues times) if attribute is nominal
        nominal_attribute_value_ranges = re.findall(r'@attribute\s+(\w+)\s+\{([^}]+)\}', str(schema.get_moa_header()))
        self.nominal_attribute_dict = {name: attribute_range.split(',')
                                       for name, attribute_range in nominal_attribute_value_ranges}
        self.attribute_pos_after_transformation, self.attribute_pos_before_transformation = {}, {}
        for a, attr in enumerate(self.attribute_list_names):
            self.attribute_pos_after_transformation[attr] = len(self.attribute_list)
            self.attribute_pos_before_transformation[attr] = a
            if schema.get_moa_header().attribute(a).isNominal():
                for _ in range(schema.get_moa_header().attribute(a).numValues()):
                    self.attribute_list.append(attr)
            else:
                self.attribute_list.append(attr)

        self.softmax = Softmax(dim=-1)
        self.font_size = 7.5      # 10 for paper
        self.data_path = "data/images_sohot"  # + "/gif"

    def get_fi_impact(self, node, x):
        if x is None:
            return ""
        w_i = self.sohot.weights.get(node.orientation_sequence)
        x_dot_w_outer = [abs(val) for val in (w_i * x)]
        x_dot_w_inner = abs(sum(x_dot_w_outer))
        percentage_feature_impact = [self.sohot.alpha * (x_dot_w_outer[j] / x_dot_w_inner).detach().item() for j in
                                     range(len(x_dot_w_outer))]
        num_features = len(w_i)
        average_percentage = 1 / num_features
        impact_str = ""

        if self.schema is not None:
            feature_one_hot_offset = 0
            for i in range(self.schema.get_num_attributes()):
                # Verify if feature is one hot encoded
                if self.schema.get_moa_header().attribute(i).isNominal() \
                        and self.schema.get_moa_header().attribute(i).numValues() > 2:
                    len_attribute_values = self.schema.get_moa_header().attribute(i).numValues()
                    start, end = i + feature_one_hot_offset, i + feature_one_hot_offset + len_attribute_values
                    impact = sum(percentage_feature_impact[start:end]) / (end - start)  # average
                    feature_one_hot_offset += len_attribute_values - 1
                else:
                    impact = percentage_feature_impact[i + feature_one_hot_offset]
                # Add feature as important if impact is high
                if impact >= average_percentage:
                    impact_str += f"\n{self.attribute_list_names[i]}: {impact:.3f}"
        return impact_str

    def get_node_label(self, feature_idx, revert_transformation, split_at_value, impact):
        feature_name = self.attribute_list[feature_idx]  # Note: this index is based on the transformed feature
        # If feature is nominal, print the attribute value of the selected one hot encoded feature
        if feature_name in self.nominal_attribute_dict:
            feature_value_idx = self.attribute_pos_after_transformation[feature_name] - feature_idx
            feature_name += f"({self.nominal_attribute_dict[feature_name][feature_value_idx]})"
        elif revert_transformation is not None:
            # Revert the transformation of the split point only for numerical attributes
            feature_pos_before = self.attribute_pos_before_transformation[feature_name]
            split_at_value = revert_transformation(split_at_value, feature_pos_before)
        return f"{feature_name} > {split_at_value:.2f}{impact}"

    # todo only for paper figure
    @staticmethod
    def format_weight_vector(arr):
        formatted_str = ', '.join(f"{x:.4f}" for x in arr)
        if len(arr) > 4:
            formatted_str = ""
            for i in range(len(arr)):
                formatted_str += f"{arr[i]:.4f}"
                if i != len(arr) - 1: formatted_str += ","
                if i % 5 == 0 and i != 0 and i != len(arr) - 1:
                    formatted_str += "\n      "
        return formatted_str

    def visualize_soft_hoeffding_tree(self, X=None, print_idx=0, save_img=False, revert_transformation=None,
                                      print_edge_labels=True):
        if save_img:
            Path(self.data_path).mkdir(parents=True, exist_ok=True)
        G = nx.Graph()
        node_label = {}
        edge_label = {}
        G.add_node(self.sohot.root)

        # --------------- Tree has only one node ---------------
        if isinstance(self.sohot.root, LeafNode):
            leaf_prob_dist = self.softmax(self.sohot.weights[self.sohot.root.orientation_sequence]).data.numpy()
            leaf_prob_dist_str = ",".join(f"{num:.2f}" for num in leaf_prob_dist)
            node_label[
                self.sohot.root] = f"P(x->l)={self.sohot.root.sample_to_node_prob:.2f}\nDist:({leaf_prob_dist_str})"
            pos = nx.spring_layout(G)
            label_shift, label_shift_leaf = 0.01, 0.02

        # --------------- Traverse whole tree ---------------
        else:
            if len(self.attribute_list) == 0:
                node_label[self.sohot.root] = "Split Attr: {}, \nValue:{:.2f}".format(
                    self.sohot.root.split_test.feature,
                    self.sohot.root.split_test.split_at)
            else:
                impact = self.get_fi_impact(self.sohot.root, X)
                node_label[self.sohot.root] = self.get_node_label(feature_idx=self.sohot.root.split_test.feature,
                                                                  revert_transformation=revert_transformation,
                                                                  split_at_value=self.sohot.root.split_test.split_at,
                                                                  impact=impact)
                # todo Only for paper figure to show before and after:
                # node_label[self.sohot.root] = f"w=({self.format_weight_vector(self.sohot.weights.get(self.sohot.root.orientation_sequence).detach().numpy())})"

            if self.sohot.root.right_leaf is None:
                to_traverse = [self.sohot.root.right]
            else:
                to_traverse = [self.sohot.root.right_leaf]
            if self.sohot.root.left_leaf is None:
                to_traverse.append(self.sohot.root.left)
            else:
                to_traverse.append(self.sohot.root.left_leaf)
            previous = [self.sohot.root, self.sohot.root]

            while to_traverse:
                i = to_traverse.pop()
                prev = previous.pop()
                G.add_node(i)
                G.add_edge(prev, i)

                if isinstance(i, Node):
                    impact = self.get_fi_impact(i, X)
                    node_label[i] = self.get_node_label(feature_idx=i.split_test.feature,
                                                        revert_transformation=revert_transformation,
                                                        split_at_value=i.split_test.split_at,
                                                        impact=impact)

                    # todo Only for paper figure to show before and after:
                    # node_label[i] = f"w=({self.format_weight_vector(self.sohot.weights.get(i.orientation_sequence).detach().numpy())})"

                    if print_edge_labels:
                        w_i = self.sohot.weights.get(prev.orientation_sequence)
                        if prev.left is i:
                            edge_label[(prev, i)] = "{:.2f}".format(prev.forward(X, w_i))
                        else:
                            edge_label[(prev, i)] = "{:.2f}".format(1. - prev.forward(X, w_i))

                    if i.right_leaf is None:
                        to_traverse.append(i.right)
                    else:
                        to_traverse.append(i.right_leaf)
                    if i.left_leaf is None:
                        to_traverse.append(i.left)
                    else:
                        to_traverse.append(i.left_leaf)
                    previous.append(i)
                    previous.append(i)
                else:
                    if print_edge_labels:
                        w_i = self.sohot.weights.get(prev.orientation_sequence)
                        if prev.left_leaf is i:
                            edge_label[(prev, i)] = "{:.2f}".format(prev.forward(X, w_i))
                        else:
                            edge_label[(prev, i)] = "{:.2f}".format(1. - prev.forward(X, w_i))
                    # Use weight to show the probability distribution of this node
                    leaf_prob_dist = self.softmax(self.sohot.weights[i.orientation_sequence]).data.numpy()
                    leaf_prob_dist_str = ",".join(f"{num:.2f}" for num in leaf_prob_dist)
                    node_label[i] = f"P(x\u2192l)={i.sample_to_node_prob:.2f}\nDist:({leaf_prob_dist_str})"

                    # todo Only for paper figure to show before and after:
                    # node_label[i] = f"P(x\u2192l)={i.sample_to_node_prob:.2f}\no=({self.format_weight_vector(self.sohot.weights.get(i.orientation_sequence).detach().numpy())})"

            pos = self._hierarchy_pos(G, self.sohot.root, width=0.5, vert_gap=0.07)   # Change for paper
            label_shift, label_shift_leaf = 0.04, 0.03
            # pos = self._hierarchy_pos(G, self.sohot.root, width=3., vert_gap=0.2)
            # label_shift, label_shift_leaf = 0.08, 0.04

        ax = plt.gca()
        if not save_img:
            ax.set_title(f"Soft Hoeffding Tree, # Instances: {print_idx}")
        nx.draw(G, pos, node_color='#8FAADC', with_labels=False)
        # Move node labels right for internal nodes and align below for leaves by label_shift
        label_pos = {node: (x + label_shift, y) if isinstance(node, Node) else (x - label_shift, y - label_shift_leaf)
                     for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels=node_label, font_size=self.font_size,
                                verticalalignment='center',
                                horizontalalignment='left',
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        if print_edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label, font_size=7)

        # Enlarge the figure to show all labels
        x_values, y_values = zip(*pos.values())
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        plt.xlim(x_min - 0.1, x_max + label_shift + 0.3)       # for paper: (..., x_max + label_shift + 0.1)
        plt.ylim(y_min - 0.1, y_max + 0.1)
        # plt.ylim(y_min - 0.1, y_max + 0.08)
        plt.margins(0.05)
        plt.tight_layout()
        if save_img:
            plt.savefig(f"{self.data_path}/figure-tree-at-{print_idx}.pdf", format="pdf")
            # plt.savefig(f"{self.data_path}/figure-tree-at-{print_idx}.png")
            plt.clf()
        else:
            plt.show()

    # source: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
    @staticmethod
    def _hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
            '''
            see hierarchy_pos docstring for most arguments

            pos: a dict saying where all nodes go if they have been assigned
            parent: parent of this branch. - only affects it if non-directed

            '''

            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                         vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                         pos=pos, parent=root)
            return pos

        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
