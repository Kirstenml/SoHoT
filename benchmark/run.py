from benchmark.evaluator import TreeEvaluator
from benchmark.load_data import load_data_stream
from benchmark.hyperparameter_tuning_pool import ModelPool
from sohot.sohot_ensemble_layer import SoftHoeffdingTreeLayer
from capymoa.classifier import HoeffdingTree, EFDT, HoeffdingAdaptiveTree, AdaptiveRandomForestClassifier, \
    StreamingRandomPatches
from capymoa.regressor import SOKNLBT, SOKNL, KNNRegressor, SGDRegressor
from capymoa.evaluation import RegressionEvaluator
from benchmark.helpers import SGDClassifierMod, HoeffdingTreeLimit, HoeffdingTreeMod, HoeffdingAdaptiveTreeMod, \
    HoeffdingTreeRiver, HoeffdingAdaptiveTreeRiver, EFDTMod, EFDTRiver, HoeffdingTreeRegressorRiver, \
    HoeffdingAdaptiveTreeRegressorRiver, SGDRegressorScale
from pathlib import Path
import pandas as pd
import yaml
import itertools
import time

# Install TEL first before running the code - see benchmark.tel_streaming
# from benchmark.tel_streaming import TreeEnsembleLayerStreaming


def set_experiments(data_name, seed=1, data_dir=".", output_path="./benchmark/data"):
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/evaluation_results").mkdir(parents=True, exist_ok=True)
    data_stream, n_instance_limit = load_data_stream(data_name, seed=seed, data_dir=data_dir)
    schema = data_stream.get_schema()
    output_path += "/evaluation_results"
    models = [
        (SoftHoeffdingTreeLayer(schema=schema, trees_num=1, lr=0.01, ssp=1., seed=seed), f"{output_path}/SoHoT"),
        (HoeffdingTree(schema=schema, random_seed=seed), f"{output_path}/HT"),
        (HoeffdingTreeLimit(schema=schema, random_seed=seed, node_limit=127), f"{output_path}/HTLimit"),
        (HoeffdingAdaptiveTree(schema=schema, random_seed=seed), f"{output_path}/HAT"),
        (EFDT(schema=schema, random_seed=seed), f"{output_path}/EFDT"),
        (SGDClassifierMod(schema=schema, loss='log_loss', random_seed=seed), f"{output_path}/SGDClassifier"),
        # (TreeEnsembleLayerStreaming(schema=schema, trees_num=1, learning_rate=0.01, smooth_step_param=1.,
        #                             measure_transparency=False), f"{output_path}/TEL")
    ]

    for model, output_path in models:
        compute_complexity = False
        run_experiments(data_stream=data_stream, data_name=data_name, model=model, n_instance_limit=n_instance_limit,
                        seed=seed, output_path=output_path, compute_complexity=compute_complexity)


def set_hyperparameter_model_pool(data_name, seed=1, data_dir=".", output_path="./benchmark/data/evaluation_results",
                                  k=5, model_names=None):
    # Read yaml file
    with open('./benchmark/parameters.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    options = []
    for name, params in config_data.items():
        if model_names is None or (model_names is not None and f"{name}_options" in model_names):
            options.append((name, params))
    # Evaluate all hyperparameter options
    output_path += "/hyperparameter_tuned_results"
    data_stream, n_instance_limit = load_data_stream(data_name, seed=seed, data_dir=data_dir)
    schema = data_stream.get_schema()
    for name, option in options:
        pool_models = []
        compute_complexity = True
        # combinations
        option = list(itertools.product(*(option.values())))
        for opt in option:
            no_model_selected = False
            if name.__eq__('sohot_options'):
                m = SoftHoeffdingTreeLayer(schema=schema, trees_num=1, max_depth=opt[0], ssp=opt[1], alpha=opt[2],
                                           seed=seed)
                output_path_c = f"{output_path}/SoHoT"
                compute_complexity = True
            elif name.__eq__('hoeffding_tree_options'):
                if data_name.__eq__('epsilon'):
                    m = HoeffdingTreeRiver(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                           leaf_prediction=opt[2], random_seed=seed)
                else:
                    m = HoeffdingTreeMod(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                         leaf_prediction=opt[2],
                                         random_seed=seed)
                output_path_c = f"{output_path}/HT"
            elif name.__eq__('hoeffding_tree_limit_options'):
                if data_name.__eq__('epsilon'):
                    m = HoeffdingTreeRiver(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                           leaf_prediction=opt[2], random_seed=seed, limit=opt[4])
                else:
                    m = HoeffdingTreeLimit(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                           leaf_prediction=opt[2], random_seed=seed, node_limit=opt[3])
                output_path_c = f"{output_path}/HT_limit"
            elif name.__eq__('EFDT_options'):
                if data_name.__eq__('epsilon'):
                    m = EFDTRiver(schema=schema, grace_period=opt[0], min_samples_reevaluate=int(opt[1]),
                                  leaf_prediction=opt[2], random_seed=seed)
                else:
                    m = EFDTMod(schema=schema, grace_period=opt[0], min_samples_reevaluate=opt[1],
                                leaf_prediction=opt[2],
                                random_seed=seed)
                output_path_c = f"{output_path}/EFDT"
            elif name.__eq__('hat_options'):
                if data_name.__eq__('epsilon'):
                    m = HoeffdingAdaptiveTreeRiver(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                                   leaf_prediction=opt[2], random_seed=seed)
                else:
                    m = HoeffdingAdaptiveTreeMod(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                                 leaf_prediction=opt[2], random_seed=seed)
                output_path_c = f"{output_path}/HAT"
                compute_complexity = True
            elif name.__eq__('sgd_clf_options'):
                m = SGDClassifierMod(schema=schema, loss=opt[0], penalty=opt[1], learning_rate=opt[2]['lr'],
                                     eta0=opt[2]['eta0'], random_seed=seed)
                output_path_c = f"{output_path}/SGDClassifier"
                compute_complexity = False
            # elif name.__eq__('tel_options'):
            #     # todo remove if tf is not installed in venv
            #     from benchmark.tel_streaming import TreeEnsembleLayerStreaming
            #     m = TreeEnsembleLayerStreaming(schema=schema, trees_num=1, depth=opt[0], smooth_step_param=opt[1],
            #                                    learning_rate=float(opt[2]), seed=seed)
            #     output_path_c = f"{output_path}/TEL"
            #     compute_complexity = False
            else:
                no_model_selected = True
                break
            pool_models.append(m)
        if no_model_selected: continue
        model = ModelPool(models=pool_models, k=k)
        run_experiments(data_stream=data_stream, data_name=data_name, model=model, n_instance_limit=n_instance_limit,
                        seed=seed, output_path=output_path_c, compute_complexity=compute_complexity)


def run_experiments(data_stream, data_name, model, n_instance_limit, seed, output_path, compute_complexity=False):
    data_stream.restart()
    tree_evaluator = TreeEvaluator(schema=data_stream.get_schema())
    i = 0
    while data_stream.has_more_instances():
        instance = data_stream.next_instance()
        # ----------- Test -----------
        y_pred = model.predict_proba(instance)
        # ----------- Train -----------
        model.train(instance)
        # ----------- Evaluate -----------
        tree_evaluator.update(y_pred=y_pred, y_target=instance.y_index,
                              tree_complexity=model.c_complexity() if compute_complexity else None)
        i += 1
        if n_instance_limit is not None and i == n_instance_limit:
            break

    if compute_complexity:
        tree_evaluator.update_complexity(model.c_complexity())
    # ----------- Print to file -----------
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    open(f"{output_path}/{data_name}_summary-seed-{seed}.csv", "w").close()
    metrics = tree_evaluator.get_evaluation()
    print(metrics)
    df = pd.DataFrame(metrics, index=[0])
    df.to_csv(f"{output_path}/{data_name}_summary-seed-{seed}.csv",
              mode="a", index=False, header=True)


def run_time_experiments(data_stream, data_name, model, n_instance_limit, seed, output_path):
    data_stream.restart()
    i = 0
    # Time measurement
    time_before_while = time.time()
    time_forward, time_backward = [], []
    while data_stream.has_more_instances():
        instance = data_stream.next_instance()
        # ----------- Test -----------
        start = time.time()
        model.predict_proba(instance)
        time_forward.append(time.time() - start)
        # ----------- Train -----------
        start = time.time()
        model.train(instance)
        time_backward.append(time.time() - start)
        i += 1
        if n_instance_limit is not None and i == n_instance_limit:
            break
    # Time measurement
    time_after_while = time.time() - time_before_while
    Path(f"{output_path}/time").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'forward': time_forward, 'backward': time_backward})
    df.to_csv(f"{output_path}/time/{data_name}_seed_{seed}_time_details.csv")
    with open(f'{output_path}/time/{data_name}_seed_{seed}_time_summary.csv', 'a') as f:
        print(f"{time_after_while}", file=f)


def plot_transparency(data_name, seed=1, data_dir="./data", n_instance_limit=None, visualize_tree_at=[],
                      save_img=False):
    data_stream, n_instances = load_data_stream(data_name, seed=seed, data_dir=data_dir)
    if n_instance_limit is not None: n_instances = n_instance_limit
    schema = data_stream.get_schema()

    model = SoftHoeffdingTreeLayer(schema=schema, seed=seed, trees_num=1, lr=0.06)  # 0.08
    i = 0
    while data_stream.has_more_instances():
        instance = data_stream.next_instance()
        model.sohots[0]._reset_all_node_to_leaf_prob()
        y_pred = model.predict_proba(instance)

        if i in visualize_tree_at:
            print(f"Prediction at {i}: {y_pred.detach().numpy()}, true label: {instance.y_index}")
            model.plot_tree(instance=instance, tree_idx=0, save_img=save_img)

        model.train(instance)
        if i == n_instances:
            break
        i += 1


# --------------------------------------- Regression ---------------------------------------

def set_regression_experiments(data_name, seed=1, data_dir=".", output_path="./benchmark/data"):
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/regression_results").mkdir(parents=True, exist_ok=True)
    data_stream, n_instance_limit = load_data_stream(data_name, seed=seed, data_dir=data_dir)

    # todo testing
    n_instance_limit = 200
    print("--- testing ---")

    schema = data_stream.get_schema()
    output_path += "/regression_results"
    models = [
        (SoftHoeffdingTreeLayer(schema=schema, trees_num=1, lr=0.01, ssp=1., seed=seed, alpha=0.3),
         f"{output_path}/SoHoT_Reg"),
        (HoeffdingTreeRegressorRiver(schema=schema), f"{output_path}/HT_Reg"),
        (HoeffdingAdaptiveTreeRegressorRiver(schema=schema), f"{output_path}/HAT_Reg"),
        (SOKNLBT(schema, random_seed=seed), f"{output_path}/SOKNLBT"),
        (SOKNL(schema, random_seed=seed), f"{output_path}/SOKNL"),
        (KNNRegressor(schema, random_seed=seed), f"{output_path}/KNNRegressor"),
        (SGDRegressor(schema, random_seed=seed), f"{output_path}/SGDRegressor"),
        (SGDRegressorScale(schema, random_seed=seed), f"{output_path}/SGDScaleRegressor"),
        # (TreeEnsembleLayerStreaming(schema=schema, trees_num=1, learning_rate=0.01, smooth_step_param=1.,
        #                             measure_transparency=False), f"{output_path}/TEL_Reg")
    ]

    for model, output_path in models:
        run_regression_experiments(data_stream=data_stream, data_name=data_name, model=model,
                                   n_instance_limit=n_instance_limit, seed=seed, output_path=output_path, compute_complexity=True)


def run_regression_experiments(data_stream, data_name, model, n_instance_limit, seed, output_path,
                               compute_complexity=False):
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    data_stream.restart()
    i = 0
    sum_tree_complexities = None
    regression_evaluator = RegressionEvaluator(schema=data_stream.get_schema())
    while data_stream.has_more_instances():
        instance = data_stream.next_instance()
        # ----------- Test -----------
        y_pred = model.predict(instance)
        # ----------- Train -----------
        model.train(instance)
        # ----------- Evaluate -----------
        regression_evaluator.update(y=instance.y_value, y_pred=y_pred)
        if compute_complexity:
            if sum_tree_complexities is None: sum_tree_complexities = model.c_complexity()
            else: sum_tree_complexities = [sum(tc) for tc in zip(sum_tree_complexities, model.c_complexity())]
        i += 1
        if n_instance_limit is not None and i == n_instance_limit:
            break

    regressor_name = output_path.split("/")[-1]
    # Print to file
    metrics = {'rmse': regression_evaluator.rmse(), 'mae': regression_evaluator.mae(), 'r2': regression_evaluator.r2(),
               'rmae': regression_evaluator.rmae()}
    if compute_complexity:
        for j, complexity in enumerate(sum_tree_complexities):
            metrics[f'Avg Tree {j} Complexity'] = complexity / i    # i = number of seen instances
    print(f"{regressor_name}: {metrics}")
    df = pd.DataFrame(metrics, index=[0])
    df.to_csv(f"{output_path}/{data_name}_summary-seed-{seed}.csv", mode="w", index=False, header=True)


def set_hyperparameter_model_pool_regression(data_name, seed=1, data_dir="./benchmark/data",
                                             output_path="./benchmark/data", k=5):
    # Read yaml file
    with open('./benchmark/parameters.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    options = []
    for name, params in config_data.items():
        options.append((name, params))
    # Evaluate all hyperparameter options
    output_path += "/hyperparameter_tuned_results_regression"
    data_stream, n_instance_limit = load_data_stream(data_name, seed=seed, data_dir=data_dir)

    schema = data_stream.get_schema()
    for name, option in options:
        no_regression_model_selected = False
        pool_models = []
        compute_complexity = False
        # combinations
        option = list(itertools.product(*(option.values())))
        for opt in option:
            if name.__eq__('sohot_options'):
                m = SoftHoeffdingTreeLayer(schema=schema, trees_num=1, max_depth=opt[0], ssp=opt[1], alpha=opt[2],
                                           seed=seed)
                output_path_c = f"{output_path}/SoHoT_Reg"
                compute_complexity = True
            elif name.__eq__('hoeffding_tree_reg_options'):
                m = HoeffdingTreeRegressorRiver(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                                leaf_prediction=opt[2], random_seed=seed)
                output_path_c = f"{output_path}/HT_Reg"
                compute_complexity = True
            elif name.__eq__('hat_reg_options'):
                m = HoeffdingAdaptiveTreeRegressorRiver(schema=schema, grace_period=opt[0], confidence=float(opt[1]),
                                                        leaf_prediction=opt[2], random_seed=seed)
                output_path_c = f"{output_path}/HAT_Reg"
                compute_complexity = True
            elif name.__eq__('sgd_reg_options'):
                # m = SGDRegressor(schema, loss=opt[0], penalty=opt[1], learning_rate=opt[2]['lr'],
                #                  eta0=opt[2]['eta0'], random_seed=seed)
                m = SGDRegressorScale(schema, loss=opt[0], penalty=opt[1], learning_rate=opt[2]['lr'],
                                      eta0=opt[2]['eta0'], random_seed=seed)
                compute_complexity = False
                output_path_c = f"{output_path}/SGD_Reg"
            elif name.__eq__('knn_reg_options'):
                m = KNNRegressor(schema, k=opt[0], median=opt[1], window_size=opt[2], random_seed=seed)
                compute_complexity = False
                output_path_c = f"{output_path}/KNN_Reg"
            # if name.__eq__('tel_options'):
            #     m = TreeEnsembleLayerStreaming(schema=schema, trees_num=1, depth=opt[0], smooth_step_param=opt[1],
            #                                    learning_rate=float(opt[2]), seed=seed)
            #     compute_complexity = False
            #     output_path_c = f"{output_path}/TEL_Reg"
            else:
                no_regression_model_selected = True
                break
            pool_models.append(m)
        if no_regression_model_selected: continue
        model = ModelPool(models=pool_models, k=k, is_target_class=False)
        run_regression_experiments(data_stream=data_stream, data_name=data_name, model=model,
                                   n_instance_limit=n_instance_limit,
                                   seed=seed, output_path=output_path_c, compute_complexity=compute_complexity)

# --------------------------------------- Ensembles ---------------------------------------

def set_ensemble(data_name, ensemble_size=10, seed=1, data_dir=".", output_path="./benchmark/data"):
    Path(f"{output_path}/ensemble_{ensemble_size}_results").mkdir(parents=True, exist_ok=True)
    data_stream, n_instance_limit = load_data_stream(data_name, seed=seed, data_dir=data_dir)
    schema = data_stream.get_schema()
    output_path += f"/ensemble_{ensemble_size}_results"
    models = [
        (SoftHoeffdingTreeLayer(schema=schema, trees_num=ensemble_size, seed=seed), f"{output_path}/SoHoT"),
        (AdaptiveRandomForestClassifier(schema=schema, ensemble_size=ensemble_size, random_seed=seed),
         f"{output_path}/ARF"),
        (StreamingRandomPatches(schema=schema, ensemble_size=ensemble_size, random_seed=seed), f"{output_path}/SRP"),
    ]
    for model, output_path in models:
        compute_complexity = output_path.endswith('SoHoT')
        run_experiments(data_stream=data_stream, data_name=data_name, model=model, n_instance_limit=n_instance_limit,
                        seed=seed, output_path=output_path, compute_complexity=compute_complexity)
