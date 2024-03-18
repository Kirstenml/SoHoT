import torch.autograd
from .internal_node import Node
from .leaf_node import LeafNode
from .sohot_helpers import soft_activation_derivative

K_TOLERANCE = 0     # 1e-10
# If probability to reach the current leaf node > EPSILON then update leaf node statistics
EPSILON = 0.25


class SoHoTFunction(torch.autograd.Function):
    @staticmethod
    def weights_to_dict(ks, vs):
        return {k: v for k, v in zip(ks, vs)}

    @staticmethod
    def forward_single_sample(x, output, model, sample_to_node_batch, weights):
        # stack for traversing nodes
        to_traverse = [model.root]
        model.root.sample_to_node_prob = 1.
        # store previous nodes of leaves to split a leaf and allocate the pointer correctly
        while to_traverse:
            i = to_traverse.pop()
            weight_vec = weights.get(i.orientation_sequence)
            if isinstance(i, Node):
                routing_left_prob = i.forward(x, weight_vec)
                # If next Node is an internal node
                if i.left_leaf is None:
                    i.left.sample_to_node_prob = i.sample_to_node_prob * routing_left_prob
                    if routing_left_prob > K_TOLERANCE:
                        to_traverse.append(i.left)
                else:
                    i.left_leaf.sample_to_node_prob = i.sample_to_node_prob * routing_left_prob
                    if routing_left_prob > K_TOLERANCE:
                        to_traverse.append(i.left_leaf)
                if i.right_leaf is None:
                    routing_right_prob = 1. - routing_left_prob
                    i.right.sample_to_node_prob = i.sample_to_node_prob * routing_right_prob
                    if routing_right_prob > K_TOLERANCE:
                        to_traverse.append(i.right)
                else:
                    routing_right_prob = 1. - routing_left_prob
                    i.right_leaf.sample_to_node_prob = i.sample_to_node_prob * routing_right_prob
                    if routing_right_prob > K_TOLERANCE:
                        to_traverse.append(i.right_leaf)
            # Process leaf nodes
            else:
                try:
                    sample_to_node_batch[i.orientation_sequence].append(i.sample_to_node_prob)
                except KeyError:
                    sample_to_node_batch[i.orientation_sequence] = [i.sample_to_node_prob]
                output = torch.add(output, i.forward(i.sample_to_node_prob, weight_vec))
        return output, sample_to_node_batch


    @staticmethod
    def forward(inputs, smooth_step_param, max_depth, model, targets, *args):
        batch_size = inputs.size(dim=0)
        # Restore ParameterDict
        ks, *vs = args
        weights = SoHoTFunction.weights_to_dict(ks, vs)
        # init tensors on right device (same as inputs)
        outputs = inputs.new_zeros(batch_size, model.output_dim)
        # Do not override sample_to_node probability with each new sample - store all for one batch
        sample_to_node_batch = {}
        for b in range(batch_size):
            outputs[b], sample_to_node_batch = SoHoTFunction.forward_single_sample(inputs[b], outputs[b], model,
                                                                                 sample_to_node_batch, weights)
        # Return sample_to_node_batch to store the intermediate result in the setup_context and use it in the backward
        return outputs, sample_to_node_batch


    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        inp, smooth_step_param, max_depth, model, targets, ks, *vs = inputs
        ctx.smooth_step_param = smooth_step_param
        ctx.max_depth = max_depth
        ctx.model = model
        ctx.targets = targets
        ctx.save_for_backward(inp, *vs)
        ctx.ks = ks
        _, ctx.sample_to_node_batch = output

    @staticmethod
    def backward(ctx, grad_output, arg):
        smooth_step_param = ctx.smooth_step_param
        max_depth = ctx.max_depth
        model = ctx.model
        targets = ctx.targets
        sample_to_node_batch = ctx.sample_to_node_batch
        inputs, *vs = ctx.saved_tensors
        ks = ctx.ks
        weights = SoHoTFunction.weights_to_dict(ks, vs)
        batch_size = inputs.size(dim=0)
        input_dim = model.input_dim
        grad_loss_wrt_x = inputs.new_zeros(batch_size, input_dim)

        grad_weights = {}
        for k in weights:
            w_vec = weights[k]
            grad_dim = w_vec.size(dim=0)
            grad_weights[k] = w_vec.new_zeros(batch_size, grad_dim)

        for batch in range(batch_size):
            x = inputs[batch]
            y = targets[batch]
            # Compute the gradients for the input and the weights
            postorder = model.postorder_traversal(x)
            grad_out = grad_output[batch]
            for node_idx, n in enumerate(postorder):
                node_weight = model.weights.get(n.orientation_sequence)
                if isinstance(n, LeafNode):
                    sample_to_node_c_leaf = sample_to_node_batch[n.orientation_sequence].pop(0)
                    grad_loss_wrt_o = torch.mul(grad_out, sample_to_node_c_leaf)
                    # Gradient dict
                    grad_weights[n.orientation_sequence][batch] = grad_loss_wrt_o
                    n.sum_g = torch.matmul(grad_loss_wrt_o, node_weight).item()

                    # update leaf node statistics if max node depth is not reached
                    if n.depth < max_depth:
                        if sample_to_node_c_leaf > EPSILON:
                            class_idx = y.item()
                            n.update_stats(class_idx)
                            # Better use x_raw to update attribute observer for transparency
                            n.update_splitters(x, class_idx)

                # process an internal node only if it is fractional (i.e., belongs to the fractional tree, see TEL)
                # 0 < n.forward(X) < 1 see TEL implementation -> neural_trees_helpers
                # Note: there will be no gradient calculation if the routing probability is 0 or 1 (fractional tree)
                elif 0 < (n_forward := n.forward(x, node_weight)) < 1:
                    activation_derivative = soft_activation_derivative(torch.matmul(node_weight, x),
                                                                       smooth_step_param)
                    mu1 = torch.div(activation_derivative, n_forward)
                    mu2 = torch.div(activation_derivative, (1. - n_forward))
                    if n.left is not None:
                        a = torch.mul(mu1, n.left.sum_g)
                        n.sum_g = n.left.sum_g
                    else:
                        a = torch.mul(mu1, n.left_leaf.sum_g)
                        n.sum_g = n.left_leaf.sum_g
                    if n.right is not None:
                        b = torch.mul(mu2, n.right.sum_g)
                        n.sum_g += n.right.sum_g
                    else:
                        b = torch.mul(mu2, n.right_leaf.sum_g)
                        n.sum_g += n.right_leaf.sum_g
                    # derivative of L wrt X (maybe very time-consuming and not necessary to compute)
                    a_minus_b = torch.sub(a, b)
                    grad_loss_wrt_x[batch] = torch.add(grad_loss_wrt_x[batch], torch.mul(node_weight, a_minus_b))
                    # derivative of L wrt internal node weight w_n
                    grad_loss_wrt_w = torch.mul(x, a_minus_b)
                    grad_weights[n.orientation_sequence][batch] = grad_loss_wrt_w
                else:
                    if n.left is not None: n.sum_g = n.left.sum_g
                    else: n.sum_g = n.left_leaf.sum_g
                    if n.right is not None: n.sum_g += n.right.sum_g
                    else: n.sum_g += n.right_leaf.sum_g
        if model.growth_allowed: SoHoTFunction._update_tree_structure(model, max_depth)
        # transform gradient dict to list
        _, grads_parameter = zip(*[(k, v) for k, v in grad_weights.items()])
        return grad_loss_wrt_x, None, None, None, None, None, *grads_parameter

    # Update SHT, i.e. try to extend the tree based on the already processed samples stored in the leaves
    @staticmethod
    def _update_tree_structure(model, max_depth):
        # iterate over all leaves but also store internal nodes
        to_traverse = [model.root]
        last_level_internal = []
        while to_traverse:
            n = to_traverse.pop()
            if isinstance(n, Node):
                if n.left_leaf is None:
                    to_traverse.append(n.left)
                else:
                    last_level_internal.append(n)
                    to_traverse.append(n.left_leaf)
                if n.right_leaf is None:
                    to_traverse.append(n.right)
                else:
                    last_level_internal.append(n)
                    to_traverse.append(n.right_leaf)
            else:
                previous = last_level_internal.pop() if last_level_internal else None
                if n.depth < max_depth:
                    weight_seen = n.total_weight
                    weight_diff = weight_seen - n.last_split_attempt_at
                    if weight_diff >= model.grace_period:
                        model.attempt_to_split(n, previous)
                        n.last_split_attempt_at = weight_seen
