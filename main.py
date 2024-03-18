from benchmark.parameter_choice import get_parameter_sohot, get_parameter_ht
from benchmark.testing import evaluate_sohotel, evaluate_ht, measure_transparency_sohot

if __name__ == '__main__':
    # Data streams 0 - 7:   No download necessary
    #              8 - 11:  Download
    #              12 - 19: Download from Penn Machine Learning Benchmark datasets and apply ctgan
    _data_stream_names = {0: 'sea50', 1: 'sea5E5',
                          2: 'hyperplane10_0001', 3: 'hyperplane10_001',
                          4: 'randomrbf50_0001', 5: 'randomrbf50_001',
                          6: 'agrawal_abrupt_drift', 7: 'agrawal_gradual_drift',
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
    #
    # # --------------------------------- SoHoT Transparency Evaluation ---------------------------------
    # measure_transparency_sohot(data_stream, nrows=nrows)
    #
    # # ---------------------------------------- HT Evaluation ----------------------------------------
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


    from benchmark.load_data import get_data_loader
    from sohot.sohot_ensemble_layer import SoftHoeffdingTreeLayer
    import torch

    lr = 1e-3
    batch_size = 32
    data, input_dim, output_dim, _ = get_data_loader('sea50', batch_size=batch_size, nrows=10000)
    sohotel = SoftHoeffdingTreeLayer(input_dim, output_dim)

    optim = torch.optim.Adam(sohotel.parameters(), lr=lr)
    optim.zero_grad()
    old_params = sum(1 for _ in sohotel.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=-1)

    num_correct = 0
    num_samples = 0

    for i, (x, y) in enumerate(data):
        output = sohotel(x, y)

        output_softmax = [softmax(out) for out in output.detach()]
        for b in range(batch_size):
            num_samples += 1
            if torch.argmax(output_softmax[b]) == y[b]:
                num_correct += 1

        # If the number of parameters has changed, 'update' the optimizer
        if old_params != sum(1 for _ in sohotel.parameters()):
            optim = torch.optim.Adam(sohotel.parameters(), lr=lr)
            old_params = sum(1 for _ in sohotel.parameters())

        loss = criterion(output, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
    accuracy = num_correct / num_samples
    print("Accuracy: {:.4f}".format(accuracy))
