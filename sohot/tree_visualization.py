# Graph Visualization of Soft Hoeffding Trees
# Bibliography: NetworkX
import networkx as nx
from .internal_node import Node
from .leaf_node import LeafNode
import matplotlib.pyplot as plt
import random

'''
Note: Edge probabilities are visualized with 3 decimals, therefore probability 1 is not always 1.
    Color: #8FAADC
'''


def visualize_soft_hoeffding_tree(sohot, X=None, print_idx=0, save_img=False, attribute_list={}):
    G = nx.Graph()
    node_label = {}
    edge_label = {}
    G.add_node(sohot.root)
    no_edge_labels = False
    if X is None:
        # do not print edge labels
        no_edge_labels = True

    if isinstance(sohot.root, LeafNode):
        node_label[sohot.root] = "w: {}, \n Pr: {:.2f}".format(
            sohot.weights[sohot.root.orientation_sequence].data.numpy(),
            sohot.root.sample_to_node_prob)
        nx.draw(G, labels=node_label, with_labels=True)
        plt.show()
        return

    if len(attribute_list) == 0:
        node_label[sohot.root] = "Split Attr: {}, \nValue:{:.3f}".format(sohot.root.split_test.feature,
                                                                         sohot.root.split_test.split_at)
    else:
        node_label[sohot.root] = "If {} > {:.3f}".format(attribute_list[sohot.root.split_test.feature],
                                                         sohot.root.split_test.split_at)
    if sohot.root.right_leaf is None:
        to_traverse = [sohot.root.right]
    else:
        to_traverse = [sohot.root.right_leaf]
    if sohot.root.left_leaf is None:
        to_traverse.append(sohot.root.left)
    else:
        to_traverse.append(sohot.root.left_leaf)
    previous = [sohot.root, sohot.root]

    while to_traverse:
        i = to_traverse.pop()
        prev = previous.pop()
        G.add_node(i)
        G.add_edge(prev, i)
        orientation_seq = prev.orientation_sequence
        weight_vec = sohot.weights[orientation_seq]

        if isinstance(i, Node):
            if len(attribute_list) == 0:
                node_label[i] = "Split Attr: {}, \nValue:{:.3f}".format(i.split_test.feature, i.split_test.split_at)
            else:
                node_label[i] = "If {} > {:.3f}".format(attribute_list[i.split_test.feature], i.split_test.split_at)
            if not no_edge_labels:
                if prev.left is i:
                    edge_label[(prev, i)] = "{:.6f}".format(prev.forward(X, weight_vec))
                else:
                    edge_label[(prev, i)] = "{:.6f}".format(1. - prev.forward(X, weight_vec))
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
            node_label[i] = "Pr: {:.2f}".format(i.sample_to_node_prob)
            if not no_edge_labels:
                if prev.left_leaf is i:
                    edge_label[(prev, i)] = "{:.6f}".format(prev.forward(X, weight_vec))
                else:
                    edge_label[(prev, i)] = "{:.6f}".format(1. - prev.forward(X, weight_vec))

    pos = hierarchy_pos(G, sohot.root)
    ax = plt.gca()
    ax.set_title("Soft Hoeffding Tree")
    nx.draw(G, pos=pos, labels=node_label, node_color='#8FAADC', font_size=7, with_labels=True, verticalalignment='top')
    if not no_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label, font_size=7)

    if save_img:
        plt.savefig("evaluation/sohot_{}".format(print_idx))
        plt.clf()
    else:
        plt.show()


# source: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
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
