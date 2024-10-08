from benchmark.parameter_choice import get_parameter_sohot, get_parameter_ht
from benchmark.testing import evaluate_sohotel, measure_transparency_sohot, evaluate_sohotel_drift_adapt
from benchmark.hyperparameter_tuning import hyperparameter_tuning_pool
from benchmark.evaluate_ht.testing_ht import evaluate_ht, hyperparameter_tuning_pool_ht

if __name__ == '__main__':
    # Data streams 0 - 7:   No download necessary
    #              8 - 11:  Download
    #              12 - 19: Download from Penn Machine Learning Benchmark datasets and apply ctgan
    _data_stream_names = {0: 'sea50', 1: 'sea5E5',
                          2: 'hyperplane10_0001', 3: 'hyperplane10_001',
                          4: 'randomrbf50_0001', 5: 'randomrbf50_001',
                          6: 'agrawal_abrupt_drift', 7: 'agrawal_perturbation',
                          8: 'poker', 9: 'covertype', 10: 'kdd99', 11: 'epsilon',
                          12: 'ann_thyroid', 13: 'sleep', 14: 'churn', 15: 'nursery',
                          16: 'twonorm', 17: 'texture', 18: 'optdigits', 19: 'satimage'
                          }

    data_stream = _data_stream_names[0]
    nrows = 5000
    seeds = [0, 1, 2, 3, 4]
    # ---------------------------------------- SoHoT Evaluation ----------------------------------------
    # soft_hoeffding_config = get_parameter_sohot(data_stream)
    # evaluate_sohotel(data_stream, soft_hoeffding_config, nrows=nrows)

    # --------------------------------- SoHoT Transparency Evaluation ---------------------------------
    # measure_transparency_sohot(data_stream, nrows=10**5)

    # --------------------------------- SoHoT Alpha Evaluation ---------------------------------
    # soft_hoeffding_config = get_parameter_sohot(data_stream)
    # alphas = [round(i * 0.1, 1) for i in range(0, 11)]
    # soft_hoeffding_config['alpha'] = alphas[0]
    # soft_hoeffding_config['trees_num'] = 1
    # measure_transparency_sohot(data_stream, alpha=alphas[alpha], nrows=None, seed=1)

    # --------------------------------- SoHoT Drift Adaption ---------------------------------
    # soft_hoeffding_config = get_parameter_sohot(data_stream)
    # soft_hoeffding_config['trees_num'] = 1
    # evaluate_sohotel_drift_adapt(data_stream, soft_hoeffding_config, nrows=10 ** 6)

    # --------------------------------- SoHoT Hyperparameter Tuning ---------------------------------
    # soft_hoeffding_config = get_parameter_sohot(data_stream)
    # hyperparameter_tuning_pool(data_stream, soft_hoeffding_config, nrows=nrows, seed=seeds[0], device='cpu',
    #                            write_eval_to_file=False)

    # --------------------------------- Hoeffding Tree Hyperparameter Tuning ---------------------------------
    # limit_n_nodes_ht = True        # limit the number of internal nodes in a Hoeffding tree
    # hoeffding_config = get_parameter_ht(data_stream, limit_n_nodes_ht)
    # hyperparameter_tuning_pool_ht(data_stream, hoeffding_config, nrows=nrows, seed=seeds[0],
    #                               write_eval_to_file=False, oversample_rate=75)

    # ---------------------------------------- HT Evaluation ----------------------------------------
    # limit_n_nodes_ht = False        # limit the number of internal nodes in a Hoeffding tree
    # hoeffding_config = get_parameter_ht(data_stream, limit_n_nodes_ht)
    # evaluate_ht(data_stream, hoeffding_config, nrows=nrows)

    # ---------------------------------------- TEL Evaluation ----------------------------------------
    # Note: TEL need to be installed manually (Change sys path to TreeEnsembleLayer in testing_tel)
    #       Works only with Tensorflow version 2.9.0
    #   See: https://github.com/google-research/google-research/tree/master/tf_trees
    #   How to evaluate:
    #   limit_depth = False
    #   evaluate_tel(data_stream, limit_depth)
    #   measure_transparency_tel(data_stream)
