import pandas as pd
from ctgan import CTGAN
import os
import random
import pmlb
import gzip
from pathlib import Path

if not os.path.exists("data"): os.makedirs("data")
if not os.path.exists("data/ctgan"): os.makedirs("data/ctgan")


# CTGAN = Conditional Tabular Generative Adversarial Network
# Idea: Introduce alternating drift and no drift for m samples:
#           Select a random but specific class 𝑐_1
#           Sample approx. oversampling_rate examples which belong to 𝑐_1 (Oversampling step)
def train_ctgan(dataset_name='sleep', n_generate=10 ** 6, epochs=100, drift=True, n_drift=10, verbose=False,
                oversample_rate=0.75, seed=42, data_dir='data', name_args=""):
    if verbose: print(
        "Generate synthetic data from {} with oversampling rate: {}".format(dataset_name, oversample_rate))
    data, discrete_col, targets = choose_data(dataset_name, data_dir)

    ctgan = CTGAN(epochs=epochs, verbose=False)
    ctgan.fit(data, discrete_col)
    random.seed(seed)

    # Inject drift
    if drift:
        len_drift = int(n_generate // n_drift)
        synthetic_data = ctgan.sample(len_drift)
        tolerance = int((oversample_rate * len_drift) // 10)
        for i in range(1, n_drift):
            if drift and i % 2 == 1:
                target_rand = targets[random.randint(0, len(targets) - 1)]
                if verbose: print("{}. Drift: Oversample target class: {}".format(i, target_rand))
                d = ctgan.sample(len_drift)
                d_target = d[d['target'] == target_rand]
                d_not_target = d[d['target'] != target_rand]
                while d_target.shape[0] < int(oversample_rate * len_drift) - tolerance:
                    resampled_d = ctgan.sample(len_drift)
                    d_target = pd.concat([d_target, resampled_d[resampled_d['target'] == target_rand]])
                    d_not_target = pd.concat([d_not_target, resampled_d[resampled_d['target'] != target_rand]])

                n_target_remaining = min(d_target.shape[0], int(oversample_rate * len_drift) + tolerance)
                n_not_target_remaining = len_drift - n_target_remaining
                d = pd.concat([d_target.iloc[:n_target_remaining, :], d_not_target.iloc[:n_not_target_remaining, :]],
                              axis=0)
                # shuffle the rows of the dataframe
                d = d.sample(frac=1)
            else:
                d = ctgan.sample(len_drift)
            synthetic_data = pd.concat([synthetic_data, d], axis=0)
    else:
        synthetic_data = ctgan.sample(n_generate)

    # Write synthetic data to file
    if drift:
        path_dir = f'{data_dir}/ctgan/seed_{seed}/oversample_{oversample_rate}'
    else:
        path_dir = f'{data_dir}/ctgan/seed_{seed}/stationary'
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    synthetic_data.to_csv(f"{path_dir}/{dataset_name}{name_args}.csv", index=False)
    if verbose:
        for t_class in targets:
            print(
                "Class: {}, frequency: {}".format(t_class, len(synthetic_data[(synthetic_data['target'] == t_class)])))


def download_data(dataset_name, data_dir, pmlb_path):
    if not os.path.isfile(pmlb_path):
        pmlb.fetch_data(dataset_name, local_cache_dir=f'{data_dir}/downloaded_datasets')
        downloaded_path = f"{data_dir}/downloaded_datasets/{dataset_name}/{dataset_name}.tsv.gz"
        with gzip.open(downloaded_path, 'rb') as z:
            with open(pmlb_path, 'wb') as output_file:
                output_file.write(z.read())


def choose_data(dataset_name, data_dir):
    pmlb_path = f"{data_dir}/downloaded_datasets/{dataset_name}.tsv"
    # Check if data is downloaded yet and if not download it
    download_data(dataset_name, data_dir, pmlb_path)
    if dataset_name.startswith('sleep'):
        pmlb_path = f"{data_dir}/downloaded_datasets/sleep.tsv"
        data = pd.read_csv(pmlb_path, sep='\t', skiprows=0, header=0)
        discrete_col = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'target']
        targets = [0, 1, 2, 3]
    elif dataset_name.__eq__('churn'):
        data = pd.read_csv(pmlb_path, sep='\t', header=0)
        discrete_col = ['state', 'account length', 'area code', 'phone number', 'international plan', 'voice mail plan',
                        'number vmail messages', 'total day calls', 'total eve calls', 'total night calls',
                        'total intl calls', 'number customer service calls', 'target']
        targets = [0, 1]
    elif dataset_name.__eq__('nursery'):
        data = pd.read_csv(pmlb_path, sep='\t', header=0)
        discrete_col = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'target']
        targets = [0, 1, 3, 4]
    elif dataset_name.__eq__('spambase'):
        data = pd.read_csv(pmlb_path, sep='\t', header=0)
        discrete_col = ['target']
        targets = [0, 1]
    elif dataset_name.__eq__('ann_thyroid'):
        data = pd.read_csv(pmlb_path, sep='\t', skiprows=0, header=0)
        discrete_col = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16',
                        'target']
        targets = [1, 2, 3]
    elif dataset_name.startswith('twonorm'):
        pmlb_path = f"{data_dir}/downloaded_datasets/twonorm.tsv"
        data = pd.read_csv(pmlb_path, sep='\t', skiprows=0, header=0)
        discrete_col = ['target']
        targets = [0, 1]
    elif dataset_name.__eq__('texture'):
        data = pd.read_csv(pmlb_path, sep='\t', skiprows=0, header=0)
        discrete_col = ['target']
        targets = [2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14]
    elif dataset_name.__eq__('optdigits'):
        data = pd.read_csv(f'{data_dir}/downloaded_datasets/optdigits.tsv', sep='\t', skiprows=0, header=0)
        discrete_col = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8', 'input9',
                        'input10', 'input11', 'input12', 'input13', 'input14', 'input15', 'input16', 'input17',
                        'input18', 'input19', 'input20', 'input21', 'input22', 'input23', 'input24', 'input25',
                        'input26', 'input27', 'input28', 'input29', 'input30', 'input31', 'input32', 'input33',
                        'input34', 'input35', 'input36', 'input37', 'input38', 'input39', 'input40', 'input41',
                        'input42', 'input43', 'input44', 'input45', 'input46', 'input47', 'input48', 'input49',
                        'input50', 'input51', 'input52', 'input53', 'input54', 'input55', 'input56', 'input57',
                        'input58', 'input59', 'input60', 'input61', 'input62', 'input63', 'input64', 'target']
        targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif dataset_name.__eq__('satimage'):
        data = pd.read_csv(pmlb_path, sep='\t', skiprows=0, header=0)
        discrete_col = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
                        'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28',
                        'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'target']
        targets = [1, 2, 3, 4, 5, 7]
    # Return also discrete_col, which includes the column names of attributes which has integer or strings as values
    return data, discrete_col, targets
