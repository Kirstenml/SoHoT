# Note: TEL need to be installed manually (Change sys path to TreeEnsembleLayer's current location)
#       Works only with Tensorflow version 2.9.0
#   See: https://github.com/google-research/google-research/tree/master/tf_trees
import sys

sys.path.insert(1, '/home/user/TreeEnsembleLayer')
from tf_trees import TEL
import tensorflow as tf
import numpy as np
from benchmark.evaluate_tel.read_data_tel import read_data_tel
from sklearn.metrics import roc_auc_score
import pandas as pd
import itertools
from river.drift import ADWIN
import random
import math

positive_class = 1
n_trainable_models = 9


def evaluate_tel(dataset_name, seed=0, limit_depth=False, trees_num=10, depth=7, learning_rate=1e-3,
                 batch_size=32, nrows=None, smooth_step_param=1.0, write_losses_to_file=True):
    if limit_depth:
        depth = 3
    x, y, output_logits_dim = read_data_tel(dataset_name, nrows=nrows, seed=seed)
    auroc, cross_entropy = call_tel(x, y, output_logits_dim, trees_num=trees_num, depth=depth,
                                    learning_rate=learning_rate, batch_size=batch_size, seed=seed,
                                    smooth_step_param=smooth_step_param, write_losses_to_file=write_losses_to_file,
                                    dataset_name=dataset_name)
    print("TEL {} with seed {} on data stream {} "
          "with current AUROC {} and ce loss {}.".format('limit' if limit_depth else '',
                                                         seed, dataset_name,
                                                         auroc,
                                                         cross_entropy),
          )


def call_tel(x, y, output_logits_dim, trees_num=10, depth=7, learning_rate=0.001, batch_size=32, seed=0,
             smooth_step_param=1.0, write_losses_to_file=True, dataset_name=''):
    tf.random.set_seed(seed)
    tree_layer = TEL(output_logits_dim=output_logits_dim, trees_num=trees_num, depth=depth,
                     smooth_step_param=smooth_step_param)
    # Construct a sequential model with batch normalization and TEL.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tree_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_func = tf.nn.softmax_cross_entropy_with_logits
    model.compile(loss=loss_func, optimizer=optimizer)

    y_true_idx = []
    y_pred_values = []
    # Average cross-entropy loss
    ce_losses = []
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        predictions = model(x_batch, training=False)
        # AUROC: Store predictions + targets
        y_true_idx_batch = [np.argmax(y_val) for y_val in y_batch]
        y_true_idx.extend(y_true_idx_batch)
        y_pred_values.extend(tf.nn.softmax(predictions))
        ce_losses.extend(loss_func(labels=y_batch, logits=predictions).numpy())
        # Runs a single gradient update on a single batch of data
        model.train_on_batch(x_batch, y_batch)

    if output_logits_dim > 2:
        auroc = roc_auc_score(y_true_idx, y_pred_values, multi_class='ovr')
    else:
        auroc = roc_auc_score(y_true_idx, np.array(y_pred_values)[:, positive_class])

    cross_entropy_avg = np.mean(ce_losses)
    if write_losses_to_file:
        write_loss_to_file(losses=ce_losses, roc_auc=auroc, cross_entropy_avg=cross_entropy_avg,
                           dataset_name='tel/' + dataset_name, seed=seed, limited_depth=(depth == 3))
    return auroc, cross_entropy_avg


def measure_transparency_tel(data_stream, trees_num=1, depth=7, learning_rate=0.001, batch_size=32, seed=0,
                             smooth_step_param=1.0, nrows=None):
    x, y, output_logits_dim = read_data_tel(data_stream, nrows=nrows, seed=seed)
    tf.random.set_seed(seed)
    tree_layer = TEL(output_logits_dim=output_logits_dim, trees_num=trees_num, depth=depth,
                     smooth_step_param=smooth_step_param)
    # Construct a sequential model with batch normalization and TEL.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tree_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_func = tf.nn.softmax_cross_entropy_with_logits
    model.compile(loss=loss_func, optimizer=optimizer)

    # Measure transparency
    relevant_features = 0
    total_num_relevant_features = 0
    num_features = 2
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        model(x_batch, training=False)
        model.train_on_batch(x_batch, y_batch)

        # Measure the length of the explanation for each sample routing
        weights = tree_layer._node_weights[0].numpy()
        num_weights = tf.shape(weights)[1]
        for x_val in x_batch:
            for c_vec in range(num_weights):
                w_i = weights[:, c_vec]
                x_dot_w_outer = [abs(valu) for valu in (x_val * w_i)]
                x_dot_w_inner = abs(np.sum(x_dot_w_outer))
                percentage_feature_impact = [x_dot_w_outer[j] / x_dot_w_inner for j in range(len(x_dot_w_outer))]
                num_features = len(w_i)
                average_percentage = 1 / num_features
                impact = sum([1 if impact >= average_percentage else 0 for impact in percentage_feature_impact])
                relevant_features += impact
                total_num_relevant_features += 1

    average_relevant_features = relevant_features / total_num_relevant_features

    print("TEL on {}: Average number of  important features: {:.4f}, "
          "Total number of features: {} \n"
          "\tAverage ratio important feature per decision rule {:.4f}".format(data_stream, average_relevant_features,
                                                                              num_features,
                                                                              average_relevant_features / num_features))


# -----------------------------------------------------------------------------------------------------------------
# --------------------------------------- Hyperparameter Tuning TEL -----------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

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


def generate_hyperparameter_pool_tel(output_logits_dim, hyperparameter_comb):
    # Pool: (TEL, ADWIN)
    pool = []
    for comb in hyperparameter_comb:
        tree_depth, smooth_step_param, learning_rate = comb
        tel = TEL(output_logits_dim=output_logits_dim, trees_num=1, depth=tree_depth,
                  smooth_step_param=smooth_step_param)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tel)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_func = tf.nn.softmax_cross_entropy_with_logits
        model.compile(loss=loss_func, optimizer=optimizer)

        pool.append([model, ADWIN()])

    return pool


def hyperparameter_tuning_pool_tel(x, y, output_logits_dim, hyperparameter_comb, batch_size, seed):
    tf.random.set_seed(seed)
    # ------------------- Generate Pool -------------------
    pool = generate_hyperparameter_pool_tel(output_logits_dim=output_logits_dim,
                                            hyperparameter_comb=hyperparameter_comb)

    loss_func = tf.nn.softmax_cross_entropy_with_logits
    y_true_idx = []
    y_pred_values = []
    # Average cross entropy loss
    ce_losses = []
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        # ---------- Predict Prep: Iterate over models in pool and sort them increasing by their estimated loss
        pool.sort(key=lambda p: p[1].estimation)
        # Train a subset of the pool after a warm-up
        trainable_pool = get_trainable_subset_of_pool(len(ce_losses), pool)
        output = [0] * len(trainable_pool)
        # ----------- Predict
        for j, model in enumerate(trainable_pool):
            output[j] = model[0](x_batch, training=False)
        pool_output = output[0]

        # ----------- AUROC: Store predictions + targets
        y_true_idx_batch = [np.argmax(y_val) for y_val in y_batch]
        y_true_idx.extend(y_true_idx_batch)
        y_pred_values.extend(tf.nn.softmax(pool_output))
        ce_losses.extend(loss_func(labels=y_batch, logits=pool_output).numpy())

        # ---------- Train: Gradient descent
        for j, model in enumerate(trainable_pool):
            # Update ADWIN with estimated loss
            loss_vals = loss_func(labels=y_batch, logits=output[j]).numpy()
            for loss_val in loss_vals:
                model[1].update(loss_val)
            model[0].train_on_batch(x_batch, y_batch)

    if output_logits_dim > 2:
        auroc = roc_auc_score(y_true_idx, y_pred_values, multi_class='ovr')
    else:
        auroc = roc_auc_score(y_true_idx, np.array(y_pred_values)[:, positive_class])

    cross_entropy_avg = np.mean(ce_losses)
    return auroc, cross_entropy_avg, ce_losses


def evaluate_pool_tel(dataset_name, seed=0, batch_size=32, nrows=None, write_losses_to_file=True):
    tree_depths = [5, 6, 7]
    smooth_step_params = [1, 10, 100]
    learning_rates = [10 ** (-2), 10 ** (-3)]
    hyperparameter_comb = list(itertools.product(tree_depths, smooth_step_params, learning_rates))
    x, y, output_logits_dim = read_data_tel(dataset_name, nrows=nrows, seed=seed)

    auroc, cross_entropy_avg, ce_losses = hyperparameter_tuning_pool_tel(x=x, y=y, output_logits_dim=output_logits_dim,
                                                                         hyperparameter_comb=hyperparameter_comb,
                                                                         batch_size=batch_size, seed=seed)

    print("TEL with seed {} on data stream {} "
          "with current AUROC {} and ce loss {}.".format(seed, dataset_name, auroc, cross_entropy_avg),
          )

    if write_losses_to_file:
        write_loss_to_file(losses=ce_losses, roc_auc=auroc, cross_entropy_avg=cross_entropy_avg,
                           dataset_name="tuning/" + dataset_name, seed=seed)


# -----------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- write to file-----------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

def write_loss_to_file(losses, roc_auc, cross_entropy_avg, dataset_name='', seed=0, limited_depth=False):
    # ---------------------- Prequential Evaluation------------------------------------------------------------------
    sample_idx = range(len(losses))
    # columns: Cross Entropy Loss using class probabilities, Fading losses
    df = pd.DataFrame({'Sample': sample_idx,
                       'CE Loss TEL': losses})
    f = open('evaluation/losses/{}_losses_tel{}_seed_{}.csv'.format(dataset_name,
                                                                    '_limit' if limited_depth else '',
                                                                    seed), "w")
    f.write('# TEL roc auc: {}, ce loss avg: {} \n'.format(roc_auc, cross_entropy_avg))
    df.to_csv(f, index=False, lineterminator='\n')
    f.close()
