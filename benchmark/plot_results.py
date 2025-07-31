import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from benchmark.run import plot_transparency
from critdd import Diagram
from scipy import stats
from itertools import combinations
from pathlib import Path


def get_df_summary(data_names, seeds, model_names, metric, evaluation_method, data_names_latex_format=None):
    if metric.__eq__('efficiency'):
        return get_df_efficiency(data_names, seeds, model_names, evaluation_method, data_names_latex_format)
    if data_names_latex_format is None:
        table = pd.DataFrame(index=data_names, columns=model_names)
    else:
        table = pd.DataFrame(index=list(data_names_latex_format.values()), columns=model_names)
    for model in model_names:
        for data_name in data_names:
            value = 0
            seeds_seen = 0
            for seed in seeds:
                file_path = f'./data/mogon_results{evaluation_method}/{model}/{data_name}_summary-seed-{seed}.csv'
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        value += df.iloc[0][metric]
                        seeds_seen += 1
                    except:
                        pass
            if seeds_seen > 0:  # Average results
                value /= seeds_seen
                if data_names_latex_format is not None:
                    data_name = data_names_latex_format[data_name.replace("_", " ")]
                table.at[data_name, model] = value
    table.index = table.index.str.replace("_", " ")
    return table


def make_latex_table(data_names, seeds, model_names, metric, evaluation_method, data_names_latex_format=None,
                     highlight_max=True):
    table = get_df_summary(data_names=data_names, seeds=seeds, model_names=model_names, metric=metric,
                           evaluation_method=evaluation_method, data_names_latex_format=data_names_latex_format)

    # Add mean ranks
    ranks = table.rank(axis=1, method='average', ascending=False if highlight_max else True)
    mean_ranks = ranks.mean()
    table.loc['Mean Rank'] = mean_ranks

    if highlight_max:
        print(table.style.highlight_max(axis=1, props="textbf:--rwrap;").format(precision=3).to_latex())
    else:
        print(table.style.highlight_min(axis=1, props="textbf:--rwrap;").format(precision=3).to_latex())


# Paired t-test
def get_significances(data_names, seeds, model_names, metric, evaluation_method, significance_level=0.05):
    if len(model_names) != 2: Exception('Perform paired t-test only for two methods!')
    model_a, model_b = {data_name: [] for data_name in data_names}, {data_name: [] for data_name in data_names}
    for i, model in enumerate(model_names):
        for data_name in data_names:
            for seed in seeds:
                file_path = f'./data/mogon_results{evaluation_method}/{model}/{data_name}_summary-seed-{seed}.csv'
                df = pd.read_csv(file_path)
                # Add efficiency value
                if metric.__eq__('efficiency'):
                    auroc_val = df.iloc[0]['Auroc']
                    if model.__eq__('TEL'):
                        node_size_val = 149.3
                    else:
                        node_size_val = np.mean(
                            [df.iloc[0][f"Avg Tree {tree_idx} Complexity"] for tree_idx in range(12)])
                    efficiency = auroc_val / np.log2(node_size_val)
                    df['efficiency'] = [efficiency]

                if i == 0:
                    model_a[data_name].append(df.iloc[0][metric])
                else:
                    model_b[data_name].append(df.iloc[0][metric])
    n_significant_better_model_a, n_significant_better_model_b = 0, 0
    for data_name in data_names:
        t_stat, p_value = stats.ttest_rel(model_a[data_name], model_b[data_name])
        if p_value < significance_level:
            if np.mean(model_a[data_name]) > np.mean(model_b[data_name]):
                n_significant_better_model_a += 1
            else:
                n_significant_better_model_b += 1
    print(
        f"Model A is significant better on {n_significant_better_model_a} datasets and model B on {n_significant_better_model_b}.")
    return n_significant_better_model_a, n_significant_better_model_b


def significances_heatmap(data_names, seeds, model_names, metric, evaluation_method, model_names_latex_format):
    fontsize = 23
    # n_significance_table = pd.DataFrame(index=model_names, columns=model_names)
    model_names_l = [model_names_latex_format[model_name] for model_name in model_names]
    n_significance_table = pd.DataFrame(index=model_names_l, columns=model_names_l)
    for model_a in model_names:  # set diagonal to zero
        n_significance_table.at[model_names_latex_format[model_a], model_names_latex_format[model_a]] = 0
    model_pairs = list(combinations(model_names, 2))
    for model_a, model_b in model_pairs:
        m_names = [model_a, model_b]
        n_significant_better_model_a, n_significant_better_model_b = get_significances(data_names, seeds, m_names,
                                                                                       metric, evaluation_method)
        n_significance_table.at[
            model_names_latex_format[model_a], model_names_latex_format[model_b]] = n_significant_better_model_a
        n_significance_table.at[
            model_names_latex_format[model_b], model_names_latex_format[model_a]] = n_significant_better_model_b

    plt.figure(figsize=(9, 7))
    n_significance_table = n_significance_table.astype(float)
    ax = sns.heatmap(n_significance_table, annot=True, linewidths=0.5, cmap="Blues",
                     annot_kws={"size": fontsize})  # cmap="coolwarm"
    plt.xticks(fontsize=fontsize, rotation=45)
    plt.yticks(fontsize=fontsize, rotation=0)
    cbar = ax.collections[0].colorbar  # Get color bar object
    cbar.ax.tick_params(labelsize=fontsize)  # Set font size for scale labels
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"data/paper-images/figure-heatmap-{metric}.pdf", format="pdf")


# https://mirkobunse.github.io/critdd/
# Install with: pip install 'critdd @ git+https://github.com/mirkobunse/critdd'
def get_cridd(data_names, seeds, model_names, metric, evaluation_method):
    df = get_df_summary(data_names=data_names, seeds=seeds, model_names=model_names, metric=metric,
                        evaluation_method=evaluation_method)
    df = df.rename_axis('dataset_name', axis=0)
    df = df.rename_axis('classifier_name', axis=1)
    df = df.astype(np.float64)

    # create a CD diagram from the Pandas DataFrame
    diagram = Diagram(
        df.to_numpy(),
        treatment_names=df.columns,
        maximize_outcome=True
    )

    # inspect average ranks and groups of statistically indistinguishable treatments
    diagram.average_ranks  # the average rank of each treatment
    diagram.get_groups(alpha=.05, adjustment="holm")

    # export the diagram to a file
    diagram.to_file(
        f"./data/paper-images/critdd-{metric}.tex",
        alpha=.05,
        adjustment="holm",
        reverse_x=True,
        # axis_options={"title": ""},
    )


def get_df_tree_size(data_names, seeds, evaluation_method, data_names_latex_format):
    model_names = ['SoHoT', 'HT', 'HAT', 'EFDT']
    table_avg = None
    for i in range(12):
        metric = f"Avg Tree {i} Complexity"
        table = get_df_summary(data_names, seeds, model_names, metric, evaluation_method, data_names_latex_format).abs()
        if table_avg is None:
            table_avg = table
        else:
            table_avg = pd.concat([table_avg, table])
    table_avg = table_avg.groupby(level=0).mean()
    # Number of nodes for TEL not measured, therefore add manually the average number
    table_avg['TEL'] = [149.3 for _ in data_names]
    return table_avg


def get_df_efficiency(data_names, seeds, model_names, evaluation_method, data_names_latex_format):
    table_tree_size = get_df_tree_size(data_names, seeds, evaluation_method, data_names_latex_format)
    table_tree_size_log = table_tree_size.applymap(np.log2)
    table_auc = get_df_summary(data_names, seeds, model_names, 'Auroc', evaluation_method, data_names_latex_format)
    table_auc = table_auc.div(table_tree_size_log)
    table_auc = table_auc[model_names]
    return table_auc


def compare_number_of_nodes(data_names, seeds, evaluation_method, data_names_latex_format):
    model_names = ['SoHoT', 'HT', 'HAT', 'EFDT']
    table_avg = get_df_tree_size(data_names, seeds, model_names, evaluation_method, data_names_latex_format)
    # Visualization: Scatter plot
    # viridis = cm.get_cmap('viridis', len(model_names) + 1)
    viridis = cm.get_cmap('tab10', len(model_names) + 1)
    for j, model_name in enumerate(model_names):
        plt.scatter(list(data_names_latex_format.values()), table_avg[model_name], color=viridis(j))
    plt.scatter(list(data_names_latex_format.values()), [150 for _ in data_names], color=viridis(4))
    model_names.append('ST')

    plt.xticks(rotation=45)
    plt.ylabel("Number of nodes")
    plt.ylim(0, 510)
    plt.legend(model_names, loc="upper right")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"data/paper-images/figure-num_nodes.pdf", format="pdf")


def get_table_performance_complexity(data_names, seeds, evaluation_method, data_names_latex_format):
    model_names = ['SoHoT', 'HT', 'HAT', 'EFDT', 'SGDClassifier', 'TEL']

    # Get Auroc table
    table_auc = get_df_summary(data_names, seeds, model_names, 'Auroc', evaluation_method)
    # Get efficiency table
    table_trade_off = get_df_efficiency(data_names, seeds, model_names, evaluation_method, None)
    table_trade_off['SGDClassifier'] = [np.nan for _ in data_names]
    table_trade_off = table_trade_off[model_names]

    print_auroc_bold = False
    for data_name in data_names:
        data_name = data_name.replace("_", " ")
        row = f"{data_names_latex_format[data_name]} & AUC"
        idx_max_auc = table_auc.idxmax(axis=1, skipna=True)
        for i, model_name in enumerate(model_names):
            if print_auroc_bold and model_name == idx_max_auc.loc[data_name]:
                row += f" & \\textbf{{{table_auc[model_name][data_name]:.3f}}}"
            else:
                row += f" & {table_auc[model_name][data_name]:.3f}"
        row += "\\\\\n"
        row += f" & Efficiency "
        idx_max_auc = table_trade_off.idxmax(axis=1, skipna=True)
        for model_name in model_names:
            if model_name.__eq__('SGDClassifier'):
                row += f" & - "
            else:
                if model_name == idx_max_auc.loc[data_name]:
                    row += f" & \\textbf{{{table_trade_off[model_name][data_name]:.3f}}}"
                else:
                    row += f" & {table_trade_off[model_name][data_name]:.3f}"
        row += "\\\\"
        print(row)

    # Add mean ranks
    def print_ranks(ranks, metric_name):
        mean_ranks = ranks.mean()
        ranks = f" & Mean rank {metric_name}"
        for mean_rank in mean_ranks:
            ranks += f" & {mean_rank:.3f}"
        ranks += " \\\\"
        print(ranks)

    print_ranks(table_auc.rank(axis=1, method='average', ascending=False), "AUC")
    print_ranks(table_trade_off.rank(axis=1, method='average', ascending=False), "efficiency")

    # print(table_trade_off.idxmax(axis=1, skipna=True))


def compare_run_times(data_name, model_names, seeds):
    file_path = "./data/evaluation_results"
    metrics = ['forward', 'backward']
    colors = ['#276ec2', '#179b25']
    for metric in metrics:
        plt.figure(figsize=(6, 3))
        plt.rcParams.update({'font.size': 13})
        for i_m, model_name in enumerate(model_names):
            df = None
            for seed in seeds:
                df_tmp = pd.read_csv(f"{file_path}/{model_name}/time/{data_name}_seed_{seed}_time_details.csv")
                if df is None:
                    df = df_tmp
                else:
                    df = (df + df_tmp) / 2
            df = df.rolling(window=500).mean()
            plt.plot(df[metric], color=colors[i_m])
        plt.legend(['SoHoT', 'ST'])
        plt.xlabel("Instance")
        plt.ylabel("Time per instance")
        plt.tight_layout()
        # plt.title(data_name)
        # plt.show()
        plt.savefig(f"data/paper-images/figure-time-{data_name}-{metric}.pdf", format="pdf")


def boxplot_run_times(data_names, seeds, data_names_latex_format, metric):
    file_path = "./data/time_measurements"
    plt.figure(figsize=(5, 5.5))
    df = pd.DataFrame(columns=['data', 'model', 'seed', 'time'])
    i = 0

    for data_name in data_names:
        for seed in seeds:
            for model_name in ['SoHoT', 'TEL']:
                df_tmp = pd.read_csv(f"{file_path}/{model_name}/time/{data_name}_seed_{seed}_time_details.csv")
                df.loc[i] = [data_names_latex_format[data_name.replace("_", " ")], model_name, seed,
                             df_tmp[metric].mean()]
                i += 1

    sns.set_theme(style="ticks")
    sns.boxplot(x='data', y='time', hue='model', palette=["#6495ED", "#9FE2BF"], data=df)

    sns.despine(offset=10)
    plt.xlabel("Data stream")
    plt.ylabel("Average time per instance")
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"data/images_sohot/paper/figure-time-{metric}.pdf", format="pdf")


if __name__ == '__main__':
    Path(f".data/paper-images").mkdir(parents=True, exist_ok=True)
    metric = 'Auroc'
    evaluation_method = "/hyperparameter_tuned_results"
    data_names = [
        'SEA50', 'SEA_5E5', 'HYP_f', 'HYP_m', 'RBF_f', 'RBF_m', 'AGR_a', 'AGR_g',
        'sleep', 'nursery', 'twonorm', 'ann_thyroid', 'satimage', 'optdigits', 'texture', 'churn',
        'poker', 'kdd99', 'covtype', 'epsilon',
    ]
    data_names_latex_format = {'SEA50': '$\\text{SEA}_f$', 'SEA 5E5': '$\\text{SEA}_m$',
                               'HYP f': '$\\text{HYP}_f$', 'HYP m': '$\\text{HYP}_m$', 'RBF f': '$\\text{RBF}_f$',
                               'RBF m': '$\\text{RBF}_m$', 'AGR a': '$\\text{AGR}_a$', 'AGR g': '$\\text{AGR}_g$',
                               'sleep': 'Sleep', 'nursery': 'Nursery', 'twonorm': 'Twonorm',
                               'ann thyroid': 'Ann-Thyroid',
                               'satimage': 'Satimage', 'optdigits': 'Optdigits', 'texture': 'Texture', 'churn': 'Churn',
                               'poker': 'Poker', 'kdd99': 'Kdd99', 'covtype': 'Covtype',
                               'epsilon': 'Epsilon'}
    seeds = [0, 1, 2, 3, 4]
    model_names_latex_format = {'SoHoT': 'SoHoT', 'HT': 'HT', 'HT_limit': '$\\text{HT}_{\\text{limit}}$', 'HAT': 'HAT',
                                'EFDT': 'EFDT', 'SGDClassifier': 'SGDClassifier', 'TEL': 'ST'}

    # Figure B2: Average number of nodes
    compare_number_of_nodes(data_names, seeds, evaluation_method, data_names_latex_format)
    #
    # # Table 3: Results in terms of AUROC and efficiency
    get_table_performance_complexity(data_names, seeds, evaluation_method, data_names_latex_format)

    # # Figure 5: Heatmap with paired t-test
    significances_heatmap(data_names, seeds, ['SoHoT', 'HT', 'HAT', 'EFDT', 'TEL'], 'efficiency', evaluation_method,
                          model_names_latex_format)
    #
    # # Figure 6, Figure B3: Critical difference diagram (efficiency, AUROC)
    get_cridd(data_names=data_names, seeds=seeds, model_names=['SoHoT', 'HT', 'HAT', 'EFDT', 'SGDClassifier', 'TEL'],
              metric=metric, evaluation_method=evaluation_method)
    get_cridd(data_names=data_names, seeds=seeds, model_names=['SoHoT', 'HT', 'HAT', 'EFDT', 'TEL'],
              metric='efficiency', evaluation_method=evaluation_method)

    # Figure 9: Transparent tree expansion
    visualize_tree_at = [2400, 2600, 4899, 4900, 5100, 7000, 7400, 7600]
    plot_transparency(data_name='AGR_small', seed=1, visualize_tree_at=visualize_tree_at, save_img=True)

    # Figure 10, B4:
    for data_name in ['RBF_f', 'SEA50']:
        compare_run_times(data_name=data_name, model_names=['SoHoT', 'TEL'], seeds=seeds)

    # Table B2: Regression
    make_latex_table(data_names=['Fried', 'HYP_reg', 'house_8l', 'bikes'], seeds=[0, 1, 2, 3, 4],
                     model_names=["SoHoT_Reg", "HT_Reg", "HAT_Reg", "KNN_Reg", "SGD_Reg"],
                     metric="rmse", evaluation_method="/hyperparameter_tuned_results_regression",
                     data_names_latex_format=None, highlight_max=False)

    # Table 4: Stationary data
    for metric in ['Auroc', 'efficiency']:
        make_latex_table(data_names=['sleep_10e7_stationary', 'twonorm_10e7_stationary'],
                         seeds=[0, 1, 2, 3, 4],
                         model_names=['SoHoT', 'HT', 'HAT', 'EFDT', 'SGDClassifier', 'TEL'],
                         metric=metric, evaluation_method="/hyperparameter_tuned_results",
                         data_names_latex_format=None, highlight_max=True)
