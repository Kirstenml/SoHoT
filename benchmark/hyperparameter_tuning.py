# Hyperparameter optimization for data streams based on the idea of
#   N. Gunasekara, H. M. Gomes, B. Pfahringer and A. Bifet,
#   "Online Hyperparameter Optimization for Streaming Neural Networks," 2022 International Joint Conference on Neural
#   Networks (IJCNN), 2022, pp. 1-9, doi: 10.1109/IJCNN55064.2022.9891953.
import torch.nn
from sklearn.metrics import roc_auc_score
from sohot.sohot_ensemble_layer import SoftHoeffdingTreeLayer
from benchmark.load_data import *
from benchmark.write_benchmarks import write_loss_to_file
import itertools
from river.drift import ADWIN
import random
import math
import numpy as np
from benchmark.evaluate_ht.read_data_ht import load_data_ht, get_output_dims
from benchmark.evaluate_ht.testing_ht import predict_ht, evaluate_outcome
from river import tree

positive_class = 1

# ------------------------------------------ TUNING SoHoT ------------------------------------------
# Select the number of models that should be updated (in the paper M)
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


# Choose initial values from Tensorflow's Adam
def init_optimizer(parameters, lr, weight_decay):
    return torch.optim.Adam(parameters, betas=(0.9, 0.999), eps=1e-7, lr=lr, weight_decay=weight_decay)


def generate_hyperparameter_pool(input_dim, output_dim, hyperparameter_comb, lr, device, weight_decay, tie_threshold,
                                 ensemble_seeds):
    pool = []
    for comb in hyperparameter_comb:
        max_depth, ssp, alpha = comb
        sohotel = SoftHoeffdingTreeLayer(input_dim, output_dim, ssp=ssp, max_depth=max_depth, trees_num=1,
                                         tie_threshold=tie_threshold, seeds=ensemble_seeds, alpha=alpha)
        sohotel = sohotel.to(device)
        optim = init_optimizer(sohotel.parameters(), lr, weight_decay)
        optim.zero_grad()
        old_params = sum(1 for _ in sohotel.parameters())
        pool.append([sohotel, optim, old_params, ADWIN()])
    return pool


def train_model_pool(trainable_pool, criterion, criterion_return, output, y, lr, weight_decay, extension_at_batch,
                     batch_idx):
    for j, model in enumerate(trainable_pool):
        sohot, optim, old_params, adwin = model
        loss = criterion(output[j], y)
        # Update ADWIN with estimated loss
        with torch.no_grad():
            loss_val_batch = criterion_return(output[j], y).detach().cpu().numpy()
            for loss_val in loss_val_batch:
                adwin.update(loss_val)
        loss.backward()
        optim.step()
        if old_params != sum(1 for _ in sohot.parameters()):
            trainable_pool[j][1] = init_optimizer(sohot.parameters(), lr, weight_decay)
            old_params = sum(1 for _ in sohot.parameters())
            extension_at_batch.append(batch_idx)
        trainable_pool[j][1].zero_grad()
        trainable_pool[j][2] = old_params
    # return trainable_pool


def evaluate_pool(data, input_dim, output_dim, hyperparameter_comb, lr, device, weight_decay, tie_threshold,
                  ensemble_seeds):
    softmax = torch.nn.Softmax(dim=-1)
    # Pools consists of: (SoHoT, Optimizer, num_parameter, ADWIN), ADWIN is to estimate loss
    pool = generate_hyperparameter_pool(input_dim, output_dim, hyperparameter_comb, lr, device, weight_decay,
                                        tie_threshold, ensemble_seeds)

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
        # ---------- Predict Prep: Iterate over models in pool and sort them increasing by their estimated loss
        # Sort pool by estimated loss using ADWIN
        pool.sort(key=lambda p: p[3].estimation)
        # Train a subset of the pool after a warm-up
        trainable_pool = get_trainable_subset_of_pool(num_samples, pool)
        # First instances have no estimated loss but adwin returns 0
        output = [0] * len(trainable_pool)
        # ----------- Predict
        for j, model in enumerate(trainable_pool):
            output[j] = model[0](X, y)
        pool_output = output[0]

        y_true_np_class_idx.extend([y_val for y_val in [y_val.cpu().numpy() for y_val in y]])
        y_pred_softmax = [softmax(out).detach().cpu().numpy() for out in pool_output]
        y_pred_np.extend(y_pred_softmax)
        for b in range(batch_size):
            if np.argmax(y_pred_softmax[b]) == y[b]:
                num_correct += 1

        # ---------- Train: Gradient descent
        train_model_pool(trainable_pool, criterion, criterion_return, output, y, lr, weight_decay,
                         extension_at_batch, i)

        losses.extend(criterion_return(pool_output, y).detach().cpu().numpy())

        if num_samples % 10**5 <= batch_size: print("Processed samples: ", num_samples)

    # Evaluate results
    accuracy = num_correct / num_samples
    if output_dim > 2:
        eval_metric = roc_auc_score(y_true_np_class_idx, y_pred_np, multi_class='ovr')
    else:
        eval_metric = roc_auc_score(y_true_np_class_idx, np.array(y_pred_np)[:, positive_class])

    evaluation_metrics = {'accuracy': accuracy, 'roc_auc': eval_metric, 'avg_cross_entropy': np.mean(losses)}
    return evaluation_metrics, losses, extension_at_batch


def hyperparameter_tuning_pool(data_stream, config, nrows=None, oversample_rate=75, seed=0, device='cpu',
                               write_eval_to_file=False):
    # Define hyperparameters to tune
    max_depths = [5, 6, 7]
    smooth_step_params = [1, 10]
    alphas = [0.2, 0.3, 0.4]
    hyperparameter_comb = list(itertools.product(max_depths, smooth_step_params, alphas))
    lr = config['lr']
    device = device
    weight_decay = config['weight_decay']
    tie_threshold = config['tie_threshold']
    ensemble_seeds = config['ensemble_seeds']
    # --------------------- Reset the data loader and set a new seed --------------------------------------------
    data, input_dim, output_dim = get_data_loader(data_stream, batch_size=config['batch_size'], nrows=nrows,
                                                  oversample_rate=oversample_rate, seed=seed)
    # --------------------- Soft Hoeffding Tree Pool -------------------------------------------------------------
    evaluation_metrics, losses, extension_at_batch = evaluate_pool(data, input_dim, output_dim, hyperparameter_comb,
                                                                   lr, device, weight_decay, tie_threshold,
                                                                   ensemble_seeds)

    # --------------------- Evaluate Results -------------------------------------------------------------
    print("SoHoT with seed {} on data stream {} "
          "with current AUROC {:.4f} and ce loss {:.4f}.".format(seed, data_stream,
                                                                 evaluation_metrics['roc_auc'],
                                                                 evaluation_metrics['avg_cross_entropy'])
          )
    if write_eval_to_file:
        write_loss_to_file(losses, accuracy=evaluation_metrics['accuracy'], auroc=evaluation_metrics['roc_auc'],
                           ce_loss_avg=evaluation_metrics['avg_cross_entropy'],
                           dataset_name='tuning/sohot_' + data_stream, seed=seed,
                           extension_at_batch=extension_at_batch, alpha=config['alpha'])
