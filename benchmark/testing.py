import torch.nn
from sklearn.metrics import roc_auc_score
from sohot.sohot_ensemble_layer import SoftHoeffdingTreeLayer
from benchmark.load_data import *
from benchmark.write_benchmarks import write_loss_to_file
from sohot.internal_node import Node
from sohot.leaf_node import LeafNode

# Area under ROC curve
positive_class = 1


def measure_transparency_sohot(data_stream, nrows=None, oversample_rate=75,
                               seed=0, lr=1e-3, ssp=1, max_depth=7, trees_num=1, weight_decay=0,
                               device="cpu", average_output=False, tie_threshold=0.05,
                               ensemble_seeds=None, alpha=0.3):
    # --------------------- Reset the data loader and set a new seed --------------------------------------------
    data, input_dim, output_dim = get_data_loader(data_stream, batch_size=32, nrows=nrows,
                                                  oversample_rate=oversample_rate, seed=seed)

    # --------------------- Iterate through internals to measure transparency ----------------------------------
    def postorder_traversal_internals(root):
        return postorder_traversal_internals_(root)

    def postorder_traversal_internals_(node):
        postorder = []
        if node is not None and isinstance(node, Node):
            if node.left is not None:
                postorder = postorder_traversal_internals_(node.left)
            if node.right is not None:
                postorder += postorder_traversal_internals_(node.right)
            postorder.append(node)
        return postorder

    # --------------------- Soft Hoeffding Tree------------------------------------------------------------------
    def init_optimizer(parameters):
        return torch.optim.Adam(parameters, betas=(0.9, 0.999), eps=1e-7, lr=lr, weight_decay=weight_decay)

    softmax = torch.nn.Softmax(dim=-1)
    sohotel = SoftHoeffdingTreeLayer(input_dim, output_dim, ssp=ssp, max_depth=max_depth, trees_num=trees_num,
                                     average_output=average_output, tie_threshold=tie_threshold, seeds=ensemble_seeds,
                                     alpha=alpha)
    sohotel = sohotel.to(device)

    optim = init_optimizer(sohotel.parameters())
    optim.zero_grad()
    old_params = sum(1 for _ in sohotel.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    y_true_np_class_idx = []
    y_pred_np = []
    extension_at_batch = []
    num_correct = 0
    num_samples = 0
    # Measure transparency
    relevant_features = 0
    total_num_decision_rules = 0
    num_features = 2
    for i, (X, y) in enumerate(data):
        X = X.to(device)
        y = y.to(device)
        batch_size = X.size(dim=0)
        num_samples += batch_size
        output = sohotel(X, y)

        y_true_np_class_idx.extend([y_val for y_val in [y_val.cpu().numpy() for y_val in y]])
        y_pred_softmax = [softmax(out).detach().cpu().numpy() for out in output]
        y_pred_np.extend(y_pred_softmax)
        for b in range(batch_size):
            if np.argmax(y_pred_softmax[b]) == y[b]:
                num_correct += 1
        loss = criterion(output, y)
        loss.backward()
        optim.step()

        if old_params != sum(1 for _ in sohotel.parameters()):
            optim = init_optimizer(sohotel.parameters())
            old_params = sum(1 for _ in sohotel.parameters())
            extension_at_batch.append(i)
        optim.zero_grad()

        # Measure the length of the explanation for each sample routing
        # Iterate through tree
        traversal = postorder_traversal_internals(sohotel.sohots[0].root)
        for x_i in X:
            for n in traversal:
                w_i = sohotel.sohots[0].weights.get(n.orientation_sequence)
                x_dot_w_outer = [abs(valu) for valu in (w_i * x_i)]
                x_dot_w_inner = abs(sum(x_dot_w_outer))
                percentage_feature_impact = [alpha * (x_dot_w_outer[j] / x_dot_w_inner) for j in
                                             range(len(x_dot_w_outer))]
                num_features = len(w_i)
                average_percentage = 1 / num_features
                impact_indicator = [1 if impact >= average_percentage else 0 for impact in percentage_feature_impact]
                impact = sum(impact_indicator)

                # Verify if important feature is similar to split criterion feature: Count the feature only as
                # important feature if it is not similar to the feature from the split criterion
                impact_split_criterion = 0
                if impact_indicator[n.split_test.feature] == 0:
                    impact_split_criterion = int(1 - alpha >= average_percentage)
                relevant_features += impact + impact_split_criterion
                total_num_decision_rules += 1

    if output_dim > 2:
        roc_auc = roc_auc_score(y_true_np_class_idx, y_pred_np, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_true_np_class_idx, np.array(y_pred_np)[:, positive_class])

    average_relevant_features = relevant_features / total_num_decision_rules
    print("SoHoT (alpha={}) on {}: Average number of  important features: {:.4f}, "
          "Total number of features: {} \n"
          "\tAverage ratio important feature per decision rule {:.4f}\n"
          "AUROC: {}".format(alpha, data_stream,
                             average_relevant_features,
                             num_features,
                             average_relevant_features / num_features,
                             roc_auc),
          )


def call_sohot(data, input_dim, output_dim, lr=1e-3, ssp=1, max_depth=7, trees_num=10, weight_decay=0, device="cpu",
               average_output=False, tie_threshold=0.05, ensemble_seeds=None, alpha=0.3):
    # Choose initial values from Tensorflow's Adam
    def init_optimizer(parameters):
        return torch.optim.Adam(parameters, betas=(0.9, 0.999), eps=1e-7, lr=lr, weight_decay=weight_decay)

    softmax = torch.nn.Softmax(dim=-1)
    sohotel = SoftHoeffdingTreeLayer(input_dim, output_dim, ssp=ssp, max_depth=max_depth, trees_num=trees_num,
                                     average_output=average_output, tie_threshold=tie_threshold, seeds=ensemble_seeds,
                                     alpha=alpha)
    sohotel = sohotel.to(device)
    optim = init_optimizer(sohotel.parameters())
    optim.zero_grad()
    old_params = sum(1 for _ in sohotel.parameters())
    # Remark: Criterion expects the target as class indices; CrossEntropyLoss = torch.nn.NLLLoss() with log_softmax,
    # i.e. criterion_nll(torch.log(softmax(output)), y) == criterion(output, y)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_return = torch.nn.CrossEntropyLoss(reduction='none')
    losses = []
    y_true_np_class_idx = []
    y_pred_np = []
    extension_at_batch = []
    num_correct = 0
    num_samples = 0
    for i, (X, y) in enumerate(data):
        X = X.to(device)
        y = y.to(device)
        batch_size = X.size(dim=0)
        num_samples += batch_size
        output = sohotel(X, y)

        y_true_np_class_idx.extend([y_val for y_val in [y_val.cpu().numpy() for y_val in y]])
        # Caution: torch.cc.CrossEntropyLoss performs internally a softmax and expects a raw input
        y_pred_softmax = [softmax(out).detach().cpu().numpy() for out in output]
        y_pred_np.extend(y_pred_softmax)
        for b in range(batch_size):
            if np.argmax(y_pred_softmax[b]) == y[b]:
                num_correct += 1
        # Gradient descent
        loss = criterion(output, y)
        loss.backward()
        optim.step()
        losses.extend(criterion_return(output, y).detach().cpu().numpy())

        # If the number of parameters has changed, 'update' the optimizer
        if old_params != sum(1 for _ in sohotel.parameters()):
            optim = init_optimizer(sohotel.parameters())
            old_params = sum(1 for _ in sohotel.parameters())
            extension_at_batch.append(i)
        optim.zero_grad()

    # Evaluate results
    accuracy = num_correct / num_samples
    if output_dim > 2:
        eval_metric = roc_auc_score(y_true_np_class_idx, y_pred_np, multi_class='ovr')
    else:
        eval_metric = roc_auc_score(y_true_np_class_idx, np.array(y_pred_np)[:, positive_class])

    evaluation_metrics = {'accuracy': accuracy, 'roc_auc': eval_metric,
                          'avg_cross_entropy': np.mean(losses),
                          'final_complexities': [t.total_node_cnt() for t in sohotel.sohots]}
    return evaluation_metrics, losses, extension_at_batch


def evaluate_sohotel(data_stream, config, nrows=None, oversample_rate=75, seed=0, device='cpu',
                     write_eval_to_file=False):
    # --------------------- Reset the data loader and set a new seed --------------------------------------------
    loader, input_dim, output_dim = get_data_loader(data_stream, batch_size=config['batch_size'],
                                                    nrows=nrows, oversample_rate=oversample_rate,
                                                    seed=seed)
    # --------------------- Soft Hoeffding Tree------------------------------------------------------------------
    evaluation_metrics, ce_losses, extension_at_batch \
        = call_sohot(loader, input_dim, output_dim, lr=config['lr'],
                     ssp=config['ssp'], max_depth=config['max_depth'],
                     trees_num=config['trees_num'], device=device,
                     weight_decay=config['weight_decay'],
                     average_output=config['average_output'],
                     tie_threshold=config['tie_threshold'],
                     ensemble_seeds=config['ensemble_seeds'],
                     alpha=config['alpha'])
    print("SoHoT (alpha={}) with seed {} on data stream {} "
          "with current AUROC {:.4f} and ce loss {:.4f}.".format(config['alpha'], seed, data_stream,
                                                                 evaluation_metrics['roc_auc'],
                                                                 evaluation_metrics['avg_cross_entropy']),
          file=open("evaluation/results.txt", 'a')
          )
    if write_eval_to_file:
        write_loss_to_file(ce_losses, accuracy=evaluation_metrics['accuracy'], auroc=evaluation_metrics['roc_auc'],
                           ce_loss_avg=evaluation_metrics['avg_cross_entropy'],
                           dataset_name='sohot_' + data_stream, seed=seed,
                           extension_at_batch=extension_at_batch, alpha=config['alpha'])


#  ---------------------------------- Test drift adaption ----------------------------------

def call_sohot_drift_adapt(data, input_dim, output_dim, lr=1e-3, ssp=1, max_depth=7, trees_num=10, weight_decay=0,
                           device="cpu", average_output=False, tie_threshold=0.05, ensemble_seeds=None, alpha=0.3,
                           data_stream_name=""):
    # Choose initial values from Tensorflow's Adam
    def init_optimizer(parameters):
        return torch.optim.Adam(parameters, betas=(0.9, 0.999), eps=1e-7, lr=lr, weight_decay=weight_decay)

    softmax = torch.nn.Softmax(dim=-1)
    sohotel = SoftHoeffdingTreeLayer(input_dim, output_dim, ssp=ssp, max_depth=max_depth, trees_num=trees_num,
                                     average_output=average_output, tie_threshold=tie_threshold, seeds=ensemble_seeds,
                                     alpha=alpha)

    def getgradientnorm(self, grad_input, grad_output):
        print("{},{}".format(grad_output[0].norm(), self.sohots[0].total_node_cnt()),
              file=open("evaluation/adaption_drift_{}.txt".format(data_stream_name), 'a')
              )

    sohotel.register_backward_hook(getgradientnorm)

    sohotel = sohotel.to(device)
    optim = init_optimizer(sohotel.parameters())
    optim.zero_grad()
    old_params = sum(1 for _ in sohotel.parameters())
    # Remark: Criterion expects the target as class indices; CrossEntropyLoss = torch.nn.NLLLoss() with log_softmax,
    # i.e. criterion_nll(torch.log(softmax(output)), y) == criterion(output, y)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_return = torch.nn.CrossEntropyLoss(reduction='none')
    losses = []
    y_true_np_class_idx = []
    y_pred_np = []
    extension_at_batch = []
    num_correct = 0
    num_samples = 0
    for i, (X, y) in enumerate(data):
        X = X.to(device)
        y = y.to(device)
        batch_size = X.size(dim=0)
        num_samples += batch_size
        output = sohotel(X, y)

        y_true_np_class_idx.extend([y_val for y_val in [y_val.cpu().numpy() for y_val in y]])
        # Caution: torch.cc.CrossEntropyLoss performs internally a softmax and expects a raw input
        y_pred_softmax = [softmax(out).detach().cpu().numpy() for out in output]
        y_pred_np.extend(y_pred_softmax)
        for b in range(batch_size):
            if np.argmax(y_pred_softmax[b]) == y[b]:
                num_correct += 1
        # Gradient descent
        loss = criterion(output, y)
        loss.backward()
        optim.step()
        losses.extend(criterion_return(output, y).detach().cpu().numpy())

        # If the number of parameters has changed, 'update' the optimizer
        if old_params != sum(1 for _ in sohotel.parameters()):
            optim = init_optimizer(sohotel.parameters())
            old_params = sum(1 for _ in sohotel.parameters())
            extension_at_batch.append(i)
        optim.zero_grad()

    # Evaluate results
    accuracy = num_correct / num_samples
    if output_dim > 2:
        eval_metric = roc_auc_score(y_true_np_class_idx, y_pred_np, multi_class='ovr')
    else:
        eval_metric = roc_auc_score(y_true_np_class_idx, np.array(y_pred_np)[:, positive_class])

    evaluation_metrics = {'accuracy': accuracy, 'roc_auc': eval_metric,
                          'avg_cross_entropy': np.mean(losses),
                          'final_complexities': [t.total_node_cnt() for t in sohotel.sohots]}
    return evaluation_metrics, losses, extension_at_batch


def evaluate_sohotel_drift_adapt(data_stream, config, nrows=None, oversample_rate=75, seed=0, device='cpu',
                                 write_eval_to_file=False):
    # --------------------- Reset the data loader and set a new seed --------------------------------------------
    loader, input_dim, output_dim = get_data_loader(data_stream, batch_size=config['batch_size'],
                                                    nrows=nrows, oversample_rate=oversample_rate,
                                                    seed=seed)
    # --------------------- Soft Hoeffding Tree------------------------------------------------------------------
    evaluation_metrics, ce_losses, extension_at_batch \
        = call_sohot_drift_adapt(loader, input_dim, output_dim, lr=config['lr'],
                                 ssp=config['ssp'], max_depth=config['max_depth'],
                                 trees_num=config['trees_num'], device=device,
                                 weight_decay=config['weight_decay'],
                                 average_output=config['average_output'],
                                 tie_threshold=config['tie_threshold'],
                                 ensemble_seeds=config['ensemble_seeds'],
                                 alpha=config['alpha'],
                                 data_stream_name=data_stream)
    print("SoHoT (alpha={}) with seed {} on data stream {} "
          "with current AUROC {:.4f} and ce loss {:.4f}.".format(config['alpha'], seed, data_stream,
                                                                 evaluation_metrics['roc_auc'],
                                                                 evaluation_metrics['avg_cross_entropy']),
          # file=open("evaluation/results.txt", 'a')
          )
    if write_eval_to_file:
        write_loss_to_file(ce_losses, accuracy=evaluation_metrics['accuracy'], auroc=evaluation_metrics['roc_auc'],
                           ce_loss_avg=evaluation_metrics['avg_cross_entropy'],
                           dataset_name='sohot_' + data_stream, seed=seed,
                           extension_at_batch=extension_at_batch, alpha=config['alpha'])
