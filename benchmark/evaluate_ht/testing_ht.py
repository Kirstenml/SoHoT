from river import tree
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from benchmark.evaluate_ht.read_data_ht import load_data_ht, get_output_dims
from benchmark.write_benchmarks import write_loss_to_file
import itertools
from river.drift import ADWIN
import math
import random


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


def evaluate_outcome(output_dim, y_proba, y, criterion):
    # Carefully place each output on the right place in the probability  vector
    y_proba_extended = np.zeros(output_dim)
    for pred_target in y_proba:
        y_proba_extended[pred_target] = y_proba[pred_target]
    # clamp: Ensure that we have no zero elements, which will cause torch.log to produce nan or inf
    y_proba_extended_torch = torch.clamp(torch.from_numpy(y_proba_extended), min=1e-9, max=1 - 1e-9)
    loss_val = criterion(torch.log(y_proba_extended_torch), torch.tensor(y, dtype=torch.long)).item()
    return loss_val, y_proba_extended


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

        loss_val, y_proba_extended = evaluate_outcome(output_dim, y_proba, y, criterion)
        losses.append(loss_val)
        y_pred_np.append(y_proba_extended)
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


def evaluate_ht(data_stream, hoeffding_config={}, oversample_rate=75, nrows=None, seed=0, write_eval_to_file=False):
    evaluation_metrics_ht, losses_ht, complexity_ht = call_ht(data_stream,
                                                              config=hoeffding_config,
                                                              oversample_rate=oversample_rate,
                                                              nrows=nrows,
                                                              seed=seed)
    print("HT{} with seed {} on data stream {} "
          "with current AUROC {:.4f} and ce loss {:.4f}.".format(
        '_limited' if hoeffding_config['num_internal_nodes'] is not None else '',
        seed, data_stream, evaluation_metrics_ht['roc_auc'], evaluation_metrics_ht['avg_cross_entropy'])
    )
    if write_eval_to_file:
        write_loss_to_file(losses_ht, accuracy=evaluation_metrics_ht['accuracy'],
                           auroc=evaluation_metrics_ht['roc_auc'],
                           ce_loss_avg=evaluation_metrics_ht['avg_cross_entropy'],
                           dataset_name='ht{}_'.format('_limited' if hoeffding_config['num_internal_nodes'] is not None
                                                       else '') + data_stream,
                           seed=seed)


# ------------------------------------------ TUNING Hoeffding Tree ------------------------------------------
n_trainable_models = 9


def get_trainable_subset_of_pool(n_samples, pool):
    n_models_best_estimated_loss = math.ceil(n_trainable_models / 2)
    n_models_random = n_trainable_models - n_models_best_estimated_loss
    if n_samples < 1000:
        trainable_pool = pool
    else:
        # Select M pools, where M = {Half selected by lowest estimated loss, other half random}
        trainable_pool = pool[:n_models_best_estimated_loss]
        trainable_pool.extend(random.sample(pool[n_models_best_estimated_loss:], n_models_random))
    return trainable_pool


def evaluate_pool_ht(data_name, config, nrows, seed, oversample_rate, hyperparameter_comb):
    # ------------ Load data
    data, nominal_attributes = load_data_ht(data_name, oversample_rate=oversample_rate, nrows=nrows, seed=seed)
    output_dim = get_output_dims(data_name)

    # ------------ Generate Pool
    # hyperparameter combination contains: (split_confidences, leaf_predictions, grace_period)
    # Pool = (HT, ADWIN)
    pool = [[tree.HoeffdingTreeClassifier(grace_period=comb[2], delta=comb[0], leaf_prediction=comb[1],
                                          nominal_attributes=nominal_attributes),
             ADWIN()]
            for comb in hyperparameter_comb]

    # Set parameter for limited HT
    ht_limit = False
    if config['num_internal_nodes'] is not None:
        ht_limit = True
        num_internal_nodes = config['num_internal_nodes']
    correct_cnt_vfdt = 0
    n_samples = 0

    losses = []
    y_true_np_class_idx = []
    y_pred_np = []
    # Fair comparison with PyTorch model
    # See: https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/other/pytorch-lossfunc-cheatsheet.md
    criterion = torch.nn.NLLLoss()
    mse = 0
    for x, y in data:
        y_true_np_class_idx.append(y)
        # ------------ Predict
        # Sort pool by estimated loss using ADWIN
        pool.sort(key=lambda p: p[1].estimation)
        # Train a subset of the pool after a warm-up
        trainable_pool = get_trainable_subset_of_pool(n_samples, pool)
        output = [0] * len(trainable_pool)
        for j, model in enumerate(trainable_pool):
            y_pred, y_proba = predict_ht(model[0], x)
            output[j] = (y_pred, y_proba)
        y_pred, y_proba = output[0]

        # -------------- Evaluate
        loss_val, y_proba_extended = evaluate_outcome(output_dim, y_proba, y, criterion)
        losses.append(loss_val)
        y_pred_np.append(y_proba_extended)
        if y == y_pred:
            correct_cnt_vfdt += 1

        # ------------ Train
        for j, model in enumerate(trainable_pool):
            model[0].learn_one(x, y)
            # Limit: Check whether the number of internal nodes is greater than the number of permitted internal nodes
            if ht_limit and (model[0]._growth_allowed and model[0].n_nodes - model[0].n_leaves >= num_internal_nodes):
                # Deactivate growth and continue to learn in the trees
                model[0]._growth_allowed = False
            # ---- Update ADWIN
            y_pred, y_proba = output[j]
            loss_val, _ = evaluate_outcome(output_dim=output_dim, y_proba=y_proba, y=y, criterion=criterion)
            model[1].update(loss_val)

        n_samples += 1

    accuracy = correct_cnt_vfdt / n_samples
    if output_dim > 2:
        roc_auc = roc_auc_score(y_true_np_class_idx, y_pred_np, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_true_np_class_idx, np.array(y_pred_np)[:, positive_class])
    evaluation_metrics = {'accuracy': accuracy, 'roc_auc': roc_auc, 'avg_mse': mse / n_samples,
                          'avg_cross_entropy': np.mean(losses)}
    return evaluation_metrics, losses


def hyperparameter_tuning_pool_ht(data, config, nrows=None, seed=0, write_eval_to_file=False, oversample_rate=75):
    # Define hyperparameters
    split_confidences = [10 ** (-6), 10 ** (-7), 10 ** (-8)]
    leaf_predictions = ['mc', 'nba']
    grace_periods = [200, 400, 600]
    hyperparameter_comb = list(itertools.product(split_confidences, leaf_predictions, grace_periods))

    # --------------------- Hoeffding Tree Pool -------------------------------------------------------------
    evaluation_metrics, losses = evaluate_pool_ht(data_name=data, config=config, nrows=nrows, seed=seed,
                                                  oversample_rate=oversample_rate,
                                                  hyperparameter_comb=hyperparameter_comb)

    # --------------------- Evaluate Results -------------------------------------------------------------
    print("HT{} with seed {} on data stream {} "
          "with current AUROC {:.4f} and ce loss {:.4f}.".format(
        '_limited' if config['num_internal_nodes'] is not None else '',
        seed, data, evaluation_metrics['roc_auc'], evaluation_metrics['avg_cross_entropy']),
    )
    if write_eval_to_file:
        write_loss_to_file(losses, accuracy=evaluation_metrics['accuracy'],
                           auroc=evaluation_metrics['roc_auc'],
                           ce_loss_avg=evaluation_metrics['avg_cross_entropy'],
                           dataset_name='tuning/ht{}_'.format('_limited' if config['num_internal_nodes'] is not None
                                                       else '') + data,
                           seed=seed)

