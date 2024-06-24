# Return default hyperparameter


def get_parameter_sohot(dataset_name):
    max_depth = 7
    ssp = 1.0
    lr = 1e-3
    batch_size = 32
    trees_num = 10
    grace_period = 600
    split_confidence = 1e-6
    weight_decay = 0
    average_output = False
    tie_threshold = 0.05
    ensemble_seeds = [s for s in range(trees_num)]
    alpha = 0.3
    if dataset_name in ['hyperplane10_0001', 'hyperplane10_001', 'randomrbf50_0001', 'randomrbf50_001',
                        'agrawal_abrupt_drift', 'agrawal_perturbation', 'agrawal_gradual_drift',
                        'agrawal_real_gradual_drift']:
        ssp = 10

    return {'max_depth': max_depth, 'ssp': ssp, 'lr': lr, 'batch_size': batch_size, 'trees_num': trees_num,
            'grace_period': grace_period, 'split_confidence': split_confidence,
            'weight_decay': weight_decay, 'average_output': average_output, 'tie_threshold': tie_threshold,
            'ensemble_seeds': ensemble_seeds, 'alpha': alpha}


def get_parameter_ht(dataset_name, limit_n_nodes_ht=False):
    grace_period = 200
    num_internal_nodes = None
    if limit_n_nodes_ht:
        # Hyperparameter tuning: Allow only (2^(7+1)-2)/2 = 127 internal nodes to have trees of similar dimension
        num_internal_nodes = 127
    max_depth = None
    split_criterion = 'info_gain'
    delta = 1e-7
    tau = 0.05
    return {'grace_period': grace_period, 'max_depth': max_depth, 'split_criterion': split_criterion, 'delta': delta,
            'tau': tau, 'num_internal_nodes': num_internal_nodes}
