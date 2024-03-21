from river import tree
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from benchmark.evaluate_ht.read_data_ht import load_data_ht, get_output_dims

# Set path to data directory
DATA_DIR = ""
# Area under ROC curve
positive_class = 1


def call_ht(dataset_name, config={}, oversample_rate=75, nrows=None, seed=0):
    data, nominal_attributes = load_data_ht(dataset_name, oversample_rate=oversample_rate, nrows=nrows, seed=seed)
    output_dim = get_output_dims(dataset_name)
    evaluation_metrics, losses_ht, complexity_ht = hoeffding_tree(data, output_dim, config=config,
                                                                  nominal_attributes=nominal_attributes)
    return evaluation_metrics, losses_ht, complexity_ht


def predict_ht(ht, x):
    # predict probability (river/tree/hoeffding_tree_classifier)
    y_proba = ht.predict_proba_one(x)
    # only before the first learn_one y_proba={}
    if y_proba == {}:
        y_proba = {0 : 1}
    y_pred = max(y_proba, key=y_proba.get)
    return y_pred, y_proba


# Note: River by design works with dicts (xi = dict(zip(dataset.feature_names, xi)))
def hoeffding_tree(data, output_dim, config={}, ht=None, verbose=False,
                   nominal_attributes=[], batch_size=32):
    # Set parameter for limited HT
    ht_limit = False
    if config['num_internal_nodes'] is not None:
        ht_limit = True
        num_internal_nodes = config['num_internal_nodes']
    correct_cnt_vfdt = 0
    n_samples = 0
    if ht is None:
        ht = tree.HoeffdingTreeClassifier(grace_period=config['grace_period'], max_depth=config['max_depth'],
                                          split_criterion=config['split_criterion'], delta=config['delta'],
                                          tau=config['tau'], nominal_attributes=nominal_attributes)
    losses = []
    losses_batch_wise = []
    complexity_ht = []
    y_true_np_class_idx = []
    y_pred_np = []
    samples_before_report_complexity = 1
    # Fair comparison with PyTorch model
    # See: https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/other/pytorch-lossfunc-cheatsheet.md
    criterion = torch.nn.NLLLoss()
    mse = 0
    for x, y in data:
        y_true_np_class_idx.append(y)
        y_pred, y_proba = predict_ht(ht, x)
        # Carefully place each output on the right place in the probability  vector
        y_proba_extended = np.zeros(output_dim)
        for pred_target in y_proba:
            y_proba_extended[pred_target] = y_proba[pred_target]
        y_pred_np.append(y_proba_extended)
        # To compute this loss a one hot encoded target variable is needed
        y_one_hot = [0.] * output_dim
        y_one_hot[y] = 1.
        # clamp: Ensure that we have no zero elements, which will cause torch.log to produce nan or inf
        y_proba_extended_torch = torch.clamp(torch.from_numpy(y_proba_extended), min=1e-9, max=1 - 1e-9)
        losses.append(criterion(torch.log(y_proba_extended_torch), torch.tensor(y, dtype=torch.long)).item())
        if y == y_pred:
            correct_cnt_vfdt += 1
        ht.learn_one(x, y)
        n_samples += 1

        # Limit: Check whether the number of internal nodes is greater than the number of permitted internal nodes
        if ht_limit and (ht._growth_allowed and ht.n_nodes - ht.n_leaves >= num_internal_nodes):
            # Deactivate growth and continue to learn in the trees
            ht._growth_allowed = False

        if samples_before_report_complexity == batch_size:
            complexity_ht.append(ht.n_nodes)
            samples_before_report_complexity = 1
        samples_before_report_complexity += 1

    # to report the same amount of tree complexity values as after every batch for SHT
    if samples_before_report_complexity != 1:
        complexity_ht.append(ht.n_nodes)
    accuracy = correct_cnt_vfdt / n_samples
    if output_dim > 2:
        roc_auc = roc_auc_score(y_true_np_class_idx, y_pred_np, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_true_np_class_idx, np.array(y_pred_np)[:, positive_class])
    evaluation_metrics = {'accuracy': accuracy, 'roc_auc': roc_auc, 'avg_mse': mse / n_samples,
                          'avg_cross_entropy': np.mean(losses), 'n_leaves': ht.n_leaves,
                          'n_inactive_leaves': ht.n_inactive_leaves}
    return evaluation_metrics, losses, complexity_ht
