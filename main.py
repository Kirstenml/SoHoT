import os
import argparse
from benchmark.run import set_experiments, set_hyperparameter_model_pool, set_regression_experiments, \
    set_hyperparameter_model_pool_regression

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


DATA_DIR = os.getenv('DATA_DIR', './data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './benchmark/data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--seed", type=int)
    parser.add_argument("-m", "--model", type=str)
    args = parser.parse_args()

    data_name, seed, model_names = 'SEA50', 0, None
    if args.dataset is not None:
        data_name = args.dataset
    if args.seed is not None:
        seed = args.seed
    if args.model is not None:
        model = [args.model]        # select only a single model for evaluation

    data_names_reg = ['Fried', 'HYP_reg', 'house_8l', 'bikes']

    # set_experiments(data_name=data_name, seed=seed, data_dir=DATA_DIR, output_path=OUTPUT_DIR)

    if data_name in data_names_reg:
        set_hyperparameter_model_pool_regression(data_name=data_name, seed=seed, data_dir=DATA_DIR,
                                                 output_path=OUTPUT_DIR)
    else:
        set_hyperparameter_model_pool(data_name, seed=seed, data_dir=DATA_DIR, output_path=OUTPUT_DIR,
                                      model_names=model_names)
