from capymoa.datasets import Electricity, Covtype, Fried
from capymoa.stream.generator import AgrawalGenerator, RandomRBFGeneratorDrift, SEA, HyperPlaneClassification, HyperPlaneRegression
from capymoa.stream import ARFFStream
import pandas as pd
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
import os
import urllib.request
from pathlib import Path
import zipfile
import gzip
import tarfile
import csv
from datetime import datetime
from benchmark.ctgan_pmlb import train_ctgan


def create_arff(csv_data_path, data_name, data_dir, feature_names=None, index_col=None, missing_value=False,
                usecols=None, header=None, nominal_attribute_idx=[], t_data_path=None, data_frame=None):
    if os.path.exists(f'{data_dir}/{data_name}.arff'):
        return
    if data_frame is not None:
        data = data_frame
    elif feature_names is None:
        data = pd.read_csv(csv_data_path, index_col=index_col, usecols=usecols)
    else:
        if csv_data_path is not None:
            data = pd.read_csv(csv_data_path, header=header, names=feature_names, index_col=index_col, usecols=usecols)
        else:   # to handle t formatted files (here: epsilon)
            x, y = [], []
            with open(t_data_path) as eps_data:
                for idx, row in enumerate(eps_data):
                    label, features = row.split(maxsplit=1)
                    x.append(list(map(float, [s.split(':')[1] for s in features.split()])))
                    y.append(0 if label.__eq__('-1') else 1)
            data = pd.DataFrame(x)
            data.columns = feature_names
            data['class'] = y
    if missing_value:
        data.replace("?", pd.NA, inplace=True)  # In data WISDM are missing values
        data = data.apply(pd.to_numeric, errors='ignore')
        for column in data.select_dtypes(include=['number']).columns:
            data[column].fillna(data[column].mean(), inplace=True)

    relation_name = data_name
    attribute_categorical = {}  # {attribute: {attribute_value: id}}
    with open(f'{data_dir}/{data_name}.arff', 'w') as f:
        f.write(f"@relation {relation_name}\n\n")

        # Write attribute information
        for i, column in enumerate(data.columns):
            if column.__eq__('class'):  # class is always categorical
                unique_values = data[column].dropna().unique()
                unique_values_str = ','.join(map(str, unique_values))
                # Labels start at 0
                f.write(f"@attribute {column} {{{unique_values_str}}}\n")
            elif data[column].dtype.kind in 'biufc' and i not in nominal_attribute_idx:  # Numeric
                f.write(f"@attribute {column} numeric\n")
            else:  # Categorical
                unique_values = data[column].dropna().unique()
                # Write numbers starting by 1 instead of actual values
                attribute_categorical[i] = {attribute_value: j + 1 for j, attribute_value in enumerate(unique_values)}
                unique_values_str = ','.join(map(str, attribute_categorical[i].values()))
                f.write(f"@attribute {column} {{{unique_values_str}}}\n")

        # Write data
        f.write("\n@data\n")
        for _, row in data.iterrows():
            row_values = []
            for i, val in enumerate(row.values):
                if (attr_dict := attribute_categorical.get(i, None)) is not None:
                    row_values.append(attr_dict[val])
                elif i == len(data.columns) - 1:
                    try:    # Class label is an integer
                        row_values.append(int(val))
                    except ValueError:  # except for string values
                        row_values.append(val)
                else:
                    row_values.append(val)
            f.write(','.join(map(str, row_values)) + '\n')


def bike_to_arff(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        attributes = [
            "@RELATION bike_data",
            "",
            "@ATTRIBUTE year NUMERIC",
            "@ATTRIBUTE month NUMERIC",
            "@ATTRIBUTE day NUMERIC",
            "@ATTRIBUTE hour NUMERIC",
            "@ATTRIBUTE minute NUMERIC",
            "@ATTRIBUTE station {metro-canal-du-midi, place-des-carmes, place-esquirol, pomme, place-jeanne-darc}",
            "@ATTRIBUTE clouds NUMERIC",
            # "@ATTRIBUTE description {light rain, scattered clouds, clear sky, broken clouds, ...}",
            "@ATTRIBUTE humidity NUMERIC",
            "@ATTRIBUTE pressure NUMERIC",
            "@ATTRIBUTE temperature NUMERIC",
            "@ATTRIBUTE wind NUMERIC",
            "@ATTRIBUTE bikes NUMERIC",
            "",
            "@DATA"
        ]

        rows = []
        for row in reader:
            # Parse datetime
            dt = datetime.strptime(row['moment'], "%Y-%m-%d %H:%M:%S")
            values = [
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                f"'{row['station']}'",  # Quote strings
                int(row['clouds']),
                # f"'{row['description']}'",
                int(row['humidity']),
                float(row['pressure']),
                float(row['temperature']),
                float(row['wind']),
                int(row['bikes'])
            ]
            rows.append(",".join(map(str, values)))

    # Write to ARFF file
    with open(output_file, 'w', encoding='utf-8') as arff:
        arff.write("\n".join(attributes) + "\n")
        arff.write("\n".join(rows))


def load_data_stream(dataset_name, data_dir="./benchmark/data", seed=1):
    Path(f'{data_dir}').mkdir(parents=True, exist_ok=True)
    Path(f'{data_dir}/downloaded_datasets').mkdir(parents=True, exist_ok=True)
    data_stream_name, data_stream, n_instance_limit = "", None, None
    n_instance_limit = 1000000  # only if not other specified
    # ------------------- Real-world data -------------------
    if dataset_name.__eq__('airlines'):
        csv_data_path = f'{data_dir}/downloaded_datasets/airlines.csv'
        if not os.path.isfile(csv_data_path):
            local_filename, _ = urllib.request.urlretrieve(
                "https://www.kaggle.com/api/v1/datasets/download/jimschacko/airlines-dataset-to-predict-a-delay",
                f'{data_dir}/downloaded_datasets/airlines.zip')
            with zipfile.ZipFile(local_filename) as z:
                z.extract('Airlines.csv', path=f'{data_dir}/downloaded_datasets')
        feature_names = ["Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length", "class"]
        create_arff(csv_data_path=f'{data_dir}/downloaded_datasets/airlines.csv', data_name='airlines',
                    data_dir=data_dir, index_col=None, feature_names=feature_names, header=0)
        data_stream = ARFFStream(path=f'{data_dir}/airlines.arff')
    elif dataset_name.__eq__('electricity'):
        n_instance_limit = None
        data_stream = Electricity(directory=data_dir)
    elif dataset_name.__eq__('WISDM'):
        data_path = f'{data_dir}/WISDM_ar_v1.1_transformed.arff'
        # Download wisdm
        if not os.path.isfile(data_path):
            local_filename, _ = urllib.request.urlretrieve(
                "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
                f'{data_dir}/downloaded_datasets/WISDM_ar_latest.tar.gz')
            with gzip.open(local_filename, 'rb') as z:
                with tarfile.open(fileobj=z, mode='r') as tar:
                    with tar.extractfile('WISDM_ar_v1.1/WISDM_ar_v1.1_transformed.arff') as source_file:
                        with open(data_path, 'wb') as output_file:
                            output_file.write(source_file.read())
        data_stream = ARFFStream(path=data_path)
    elif dataset_name.__eq__('covtype'):
        n_instance_limit = None
        data_stream = Covtype(directory=data_dir)
    elif dataset_name.__eq__('epsilon'):
        n_instance_limit = None
        # todo add download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
        feature_names = [f'F{i}' for i in range(2000)]
        create_arff(csv_data_path=None, data_dir=data_dir, data_name='epsilon', feature_names=feature_names,
                    t_data_path=f"{data_dir}/downloaded_datasets/epsilon_normalized.t")
        data_stream = ARFFStream(path=f'{data_dir}/epsilon.arff')
    elif dataset_name.__eq__('poker'):
        n_instance_limit = None
        # todo add download from https://archive.ics.uci.edu/ml/datasets/Poker+Hand
        # Prepare dataframe because we concatenate two
        data1 = pd.read_csv(f'{data_dir}/downloaded_datasets/poker/poker-hand-testing.data', sep=',', header=None)
        data2 = pd.read_csv(f'{data_dir}/downloaded_datasets/poker/poker-hand-training-true.data', sep=',', header=None)
        data = pd.concat([data1, data2])
        data.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'class']
        create_arff(csv_data_path=None, data_dir=data_dir, data_name='poker', data_frame=data)
        data_stream = ARFFStream(path=f'{data_dir}/poker.arff')
    elif dataset_name.startswith('kdd99'):
        n_instance_limit = None
        feature_names = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
            "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
            "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class"
        ]
        csv_data_path = f'{data_dir}/downloaded_datasets/kddcup.data.corrected'
        if not os.path.isfile(csv_data_path):
            local_filename, _ = urllib.request.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz",
                                                           f'{data_dir}/downloaded_datasets/kddcup.gz')
            with gzip.open(local_filename, 'rb') as z:
                with open(f'{data_dir}/downloaded_datasets/kddcup.data.corrected', 'wb') as output_file:
                    output_file.write(z.read())
        create_arff(csv_data_path=csv_data_path, data_name='kdd99',
                    data_dir=data_dir, feature_names=feature_names, index_col=None, header=None)
        data_stream = ARFFStream(path=f'{data_dir}/kdd99.arff')
        if dataset_name.__eq__('kdd99_small'):
            n_instance_limit = 2500000
    # ------------------- CTGAN with real-world data -------------------
    elif dataset_name.startswith('sleep'):      # sleep and sleep_stationary
        if dataset_name.__eq__('sleep'):
            csv_data_path = f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/{dataset_name}.csv'
            data_name_arff = f'sleep_oversample_0.75_seed_{seed}'
        else:
            csv_data_path = f'{data_dir}/ctgan/seed_{seed}/stationary/{dataset_name[:10]}.csv'
            data_name_arff = f'{dataset_name}_seed_{seed}'

        print(f'ARFF file name: {data_dir}/{data_name_arff}')
        if not os.path.isfile(f'{data_dir}/{data_name_arff}.arff'):
            if not os.path.isfile(csv_data_path):   # Train CTGAN
                if dataset_name.__eq__('sleep'):
                    train_ctgan(dataset_name=dataset_name, n_generate=n_instance_limit, seed=seed, data_dir=data_dir)
                elif dataset_name.__eq__('sleep_10e6_stationary'):
                    train_ctgan(dataset_name='sleep', n_generate=10**6, seed=seed, data_dir=data_dir, epochs=50,
                                drift=False, name_args='_10e6')
                elif dataset_name.__eq__('sleep_10e7_stationary'):
                    train_ctgan(dataset_name='sleep', n_generate=10**7, seed=seed, data_dir=data_dir, epochs=50,
                                drift=False, name_args='_10e7')
            # Make arff file and then use ARFFStream to reach the correct schema
            feature_names = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'class']
            create_arff(csv_data_path=csv_data_path, data_dir=data_dir, data_name=data_name_arff,
                        feature_names=feature_names, header=0, nominal_attribute_idx=[3])
        data_stream = ARFFStream(path=f'{data_dir}/{data_name_arff}.arff')
    elif dataset_name.__eq__('ann_thyroid'):
        csv_data_path = f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/{dataset_name}.csv'
        if not os.path.isfile(csv_data_path):   # Train CTGAN
            train_ctgan(dataset_name=dataset_name, n_generate=n_instance_limit, seed=seed, data_dir=data_dir)
        feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
                         'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'class']
        create_arff(csv_data_path=f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/ann_thyroid.csv', data_dir=data_dir,
                    data_name=f'ann_thyroid_oversample_0.75_seed_{seed}', feature_names=feature_names, header=0)
        data_stream = ARFFStream(path=f'{data_dir}/ann_thyroid_oversample_0.75_seed_{seed}.arff')
    elif dataset_name.__eq__('churn'):
        csv_data_path = f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/{dataset_name}.csv'
        if not os.path.isfile(csv_data_path):   # Train CTGAN
            train_ctgan(dataset_name=dataset_name, n_generate=n_instance_limit, seed=seed, data_dir=data_dir)
        feature_names = [
            'state', 'account length', 'area code', 'phone number', 'international plan',
            'voice mail plan', 'number vmail messages', 'total day minutes', 'total day calls',
            'total day charge', 'total eve minutes', 'total eve calls', 'total eve charge',
            'total night minutes', 'total night calls', 'total night charge',
            'total intl minutes', 'total intl calls', 'total intl charge',
            'number customer service calls', 'class'
        ]
        create_arff(csv_data_path=f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/churn.csv', data_dir=data_dir,
                    data_name=f'churn_oversample_0.75_seed_{seed}', feature_names=feature_names, header=0,
                    nominal_attribute_idx=[0, 2])
        data_stream = ARFFStream(path=f'{data_dir}/churn_oversample_0.75_seed_{seed}.arff')
    elif dataset_name.__eq__('nursery'):
        csv_data_path = f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/{dataset_name}.csv'
        if not os.path.isfile(csv_data_path):   # Train CTGAN
            train_ctgan(dataset_name=dataset_name, n_generate=n_instance_limit, seed=seed, data_dir=data_dir)
        # finance (boolean), all other features are ordinal (they have an ordering), i. e., no nominal attributes
        feature_names = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
        create_arff(csv_data_path=f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/nursery.csv', data_dir=data_dir,
                    data_name=f'nursery_oversample_0.75_seed_{seed}', feature_names=feature_names, header=0)
        data_stream = ARFFStream(path=f'{data_dir}/nursery_oversample_0.75_seed_{seed}.arff')
    elif dataset_name.startswith('twonorm'):
        if dataset_name.__eq__('twonorm'):
            csv_data_path = f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/{dataset_name}.csv'
            data_name_arff = f'twonorm_oversample_0.75_seed_{seed}'
        else:
            csv_data_path = f'{data_dir}/ctgan/seed_{seed}/stationary/{dataset_name[:12]}.csv'
            data_name_arff = f'{dataset_name}_seed_{seed}'

        if not os.path.isfile(f'{data_dir}/{data_name_arff}.arff'):
            if not os.path.isfile(csv_data_path):   # Train CTGAN
                if dataset_name.__eq__('twonorm'):
                    train_ctgan(dataset_name=dataset_name, n_generate=n_instance_limit, seed=seed, data_dir=data_dir)
                elif dataset_name.__eq__('twonorm_10e6_stationary'):
                    train_ctgan(dataset_name='twonorm', n_generate=10 ** 6, seed=seed, data_dir=data_dir, epochs=50,
                                drift=False, name_args='_10e6')
                elif dataset_name.__eq__('twonorm_10e7_stationary'):
                    train_ctgan(dataset_name='twonorm', n_generate=10 ** 7, seed=seed, data_dir=data_dir, epochs=50,
                                drift=False, name_args='_10e7')
            feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
                             'A16', 'A17', 'A18', 'A19', 'A20', 'class']
            create_arff(csv_data_path=csv_data_path, data_dir=data_dir, data_name=data_name_arff,
                        feature_names=feature_names, header=0)
        data_stream = ARFFStream(path=f'{data_dir}/{data_name_arff}.arff')
    elif dataset_name.__eq__('optdigits'):
        csv_data_path = f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/{dataset_name}.csv'
        if not os.path.isfile(csv_data_path):   # Train CTGAN
            train_ctgan(dataset_name=dataset_name, n_generate=n_instance_limit, seed=seed, data_dir=data_dir)
        feature_names = [
            'input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8',
            'input9', 'input10', 'input11', 'input12', 'input13', 'input14', 'input15', 'input16',
            'input17', 'input18', 'input19', 'input20', 'input21', 'input22', 'input23', 'input24',
            'input25', 'input26', 'input27', 'input28', 'input29', 'input30', 'input31', 'input32',
            'input33', 'input34', 'input35', 'input36', 'input37', 'input38', 'input39', 'input40',
            'input41', 'input42', 'input43', 'input44', 'input45', 'input46', 'input47', 'input48',
            'input49', 'input50', 'input51', 'input52', 'input53', 'input54', 'input55', 'input56',
            'input57', 'input58', 'input59', 'input60', 'input61', 'input62', 'input63', 'input64',
            'class'
        ]
        create_arff(csv_data_path=f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/optdigits.csv', data_dir=data_dir,
                    data_name=f'optdigits_oversample_0.75_seed_{seed}', feature_names=feature_names, header=0)
        data_stream = ARFFStream(path=f'{data_dir}/optdigits_oversample_0.75_seed_{seed}.arff')
    elif dataset_name.__eq__('texture'):
        csv_data_path = f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/{dataset_name}.csv'
        if not os.path.isfile(csv_data_path):   # Train CTGAN
            train_ctgan(dataset_name=dataset_name, n_generate=n_instance_limit, seed=seed, data_dir=data_dir)
        feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                         'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
                         'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30',
                         'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40',
                         'class']
        create_arff(csv_data_path=f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/texture.csv', data_dir=data_dir,
                    data_name=f'texture_oversample_0.75_seed_{seed}', feature_names=feature_names, header=0)
        data_stream = ARFFStream(path=f'{data_dir}/texture_oversample_0.75_seed_{seed}.arff')
    elif dataset_name.__eq__('satimage'):
        csv_data_path = f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/{dataset_name}.csv'
        if not os.path.isfile(csv_data_path):   # Train CTGAN
            train_ctgan(dataset_name=dataset_name, n_generate=n_instance_limit, seed=seed, data_dir=data_dir)
        feature_names = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
            'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
            'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30',
            'A31', 'A32', 'A33', 'A34', 'A35', 'A36',
            'class'
        ]
        create_arff(csv_data_path=f'{data_dir}/ctgan/seed_{seed}/oversample_0.75/satimage.csv', data_dir=data_dir,
                    data_name=f'satimage_oversample_0.75_seed_{seed}', feature_names=feature_names, header=0)
        data_stream = ARFFStream(path=f'{data_dir}/satimage_oversample_0.75_seed_{seed}.arff')

    # ------------------- Drifts -------------------
    # Set a max_instance limit to 1,000,000
    # Window-size for abrupt changes cannot be set in Capymoa and are 0 as default
    elif dataset_name.__eq__('AGR_a'):
        data_stream = DriftStream(stream=[AgrawalGenerator(classification_function=2, instance_random_seed=seed),
                                          AbruptDrift(position=250000),
                                          AgrawalGenerator(classification_function=4, instance_random_seed=seed),
                                          AbruptDrift(position=500000),
                                          AgrawalGenerator(classification_function=6, instance_random_seed=seed),
                                          AbruptDrift(position=750000),
                                          AgrawalGenerator(classification_function=8, instance_random_seed=seed),
                                          ])
    elif dataset_name.__eq__('AGR_g'):
        data_stream = DriftStream(stream=[AgrawalGenerator(classification_function=2, instance_random_seed=seed),
                                          GradualDrift(position=250000, width=50000),
                                          AgrawalGenerator(classification_function=4, instance_random_seed=seed),
                                          GradualDrift(position=500000, width=50000),
                                          AgrawalGenerator(classification_function=6, instance_random_seed=seed),
                                          GradualDrift(position=750000, width=50000),
                                          AgrawalGenerator(classification_function=8, instance_random_seed=seed),
                                          ])
    elif dataset_name.__eq__('AGR_small'):
        n_instance_limit = 10000
        data_stream = DriftStream(stream=[AgrawalGenerator(classification_function=2, instance_random_seed=seed),
                                          AbruptDrift(position=2500),
                                          AgrawalGenerator(classification_function=3, instance_random_seed=seed),
                                          AbruptDrift(position=5000),
                                          AgrawalGenerator(classification_function=5, instance_random_seed=seed),
                                          AbruptDrift(position=7500),
                                          AgrawalGenerator(classification_function=6, instance_random_seed=seed),
                                          ])
    elif dataset_name.__eq__('RBF_f'):
        data_stream = RandomRBFGeneratorDrift(number_of_classes=5, number_of_attributes=10,
                                              number_of_centroids=50, number_of_drifting_centroids=50,
                                              magnitude_of_change=0.001, instance_random_seed=seed)
    elif dataset_name.__eq__('RBF_m'):
        data_stream = RandomRBFGeneratorDrift(number_of_classes=5, number_of_attributes=10,
                                              number_of_centroids=50, number_of_drifting_centroids=50,
                                              magnitude_of_change=0.0001, instance_random_seed=seed)
    elif dataset_name.startswith('SEA'):
        if dataset_name.__eq__('SEA_50'):
            width = 50
        else:
            width = 5 * (10 ** 5)
        widths = [width * i for i in range(n_instance_limit // width)]
        clf_funcs = [i % 5 for i in [1, 2, 3, 4] * (n_instance_limit // 4)]
        stream = []
        clf_func_last = 1
        for clf_func, w in zip(clf_funcs, widths):
            stream.extend([SEA(function=clf_func, instance_random_seed=seed), AbruptDrift(position=w)])
            clf_func_last = clf_func
        clf_func_next = 1 if clf_func_last == 4 else (clf_func_last + 1) % 5
        stream.append(SEA(function=clf_func_next, instance_random_seed=seed))
        data_stream = DriftStream(stream=stream)
    elif dataset_name.__eq__('HYP_f'):
        data_stream = HyperPlaneClassification(instance_random_seed=seed, number_of_classes=2, number_of_attributes=10,
                                               number_of_drifting_attributes=10, magnitude_of_change=0.001)
    elif dataset_name.__eq__('HYP_m'):
        data_stream = HyperPlaneClassification(instance_random_seed=seed, number_of_classes=2, number_of_attributes=10,
                                               number_of_drifting_attributes=10, magnitude_of_change=0.0001)

    # ------------------- Regression -------------------
    elif dataset_name.__eq__('Fried'):
        data_stream = Fried(directory=data_dir)     # + "/downloaded_datasets"
    elif dataset_name.__eq__('HYP_reg'):
        data_stream = HyperPlaneRegression(instance_random_seed=seed, number_of_classes=2, number_of_attributes=10,
                                           number_of_drifting_attributes=10, magnitude_of_change=0.0001)
    elif dataset_name.__eq__('house_8l'):
        data_stream = ARFFStream(path=f'{data_dir}/dataset_2204_house_8L.arff')
    elif dataset_name.__eq__('bikes'):
        # Convert year, month, day, hour, minute, second (leave out) to attributes
        data_path = f'{data_dir}/bikes.arff'
        if not os.path.isfile(data_path):
            # todo add automatic download: https://maxhalford.github.io/files/datasets/toulouse_bikes.zip
            bike_to_arff(f"{data_dir}/downloaded_datasets/toulouse_bikes.csv", data_path)
        data_stream = ARFFStream(path=data_path)

    return data_stream, n_instance_limit
