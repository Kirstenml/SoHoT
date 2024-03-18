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

positive_class = 1


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
    # Average cross entropy loss
    ce_losses = []
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')
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
                           dataset_name='tel/' + dataset_name,
                           seed=seed, limited_depth=(depth == 3))
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
                average_percentage = 1/num_features
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
# ---------------------------------------------------- write to file-----------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

def write_loss_to_file(losses, roc_auc, cross_entropy_avg, dataset_name='', seed=0, window_size=200, limited_depth=False):

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
