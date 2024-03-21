import pandas as pd
import numpy as np
from river import datasets
import random

DATA_DIR = ""


def one_hot_encoding_target(y, output_logits_dim):
    y_one_hot = []
    for y_i in y:
        y_vec = [0] * output_logits_dim
        y_vec[int(y_i)] = 1
        y_one_hot.append(y_vec)
    return y_one_hot


def transform_river_dataset_to_array(data, output_dim):
    x, y = [], []
    for i, (x_i, y_i) in enumerate(data):
        x_vec = list(x_i.values())
        y_vec = [0] * output_dim
        y_vec[int(y_i)] = 1
        x.append(x_vec)
        y.append(y_vec)
    return x, y


def read_data_tel(dataset_name, nrows=None, oversample_rate=75, seed=42):
    if dataset_name.startswith('sea'):
        if nrows is None: nrows = 10 ** 6
        if dataset_name[3:].__eq__('50'):
            noise = 50
        else:
            noise = 5 * (10 ** 5)
        dataset = datasets.synth.SEA(noise=noise, seed=seed)
        output_logits_dim = 2
        x, y = transform_river_dataset_to_array(dataset.take(nrows), output_dim=output_logits_dim)
    elif dataset_name.startswith('hyperplane'):
        if nrows is None: nrows = 10 ** 7
        a, b = dataset_name[10:].split('_')
        n_drift_features = int(a)
        mag_change = float('0.' + b)
        dataset = datasets.synth.Hyperplane(seed=seed, n_features=10, n_drift_features=n_drift_features,
                                            mag_change=mag_change)
        output_logits_dim = 2
        x, y = transform_river_dataset_to_array(dataset.take(nrows), output_dim=output_logits_dim)
    elif dataset_name.startswith('randomrbf'):
        if nrows is None: nrows = 10 ** 7
        a, b = dataset_name[9:].split('_')
        n_drift_centroids = int(a)
        change_speed = float('0.' + b)
        dataset = datasets.synth.RandomRBFDrift(seed_model=seed, seed_sample=seed,
                                                n_features=10, n_classes=5, n_centroids=50,
                                                change_speed=change_speed,
                                                n_drift_centroids=n_drift_centroids)
        output_logits_dim = 5
        x, y = transform_river_dataset_to_array(dataset.take(nrows), output_dim=output_logits_dim)
    elif dataset_name.startswith('agrawal'):
        num_drifts = 9
        if dataset_name.__eq__('agrawal_real_gradual_drift'):
            if nrows is None: nrows = 10 ** 7
            data = []
            position = nrows // (num_drifts + 1)  # default: 10**6
            width = position // 5
            for classification_function in range(num_drifts):
                dataset = datasets.synth.ConceptDriftStream(
                    stream=datasets.synth.Agrawal(seed=seed, classification_function=classification_function),
                    drift_stream=datasets.synth.Agrawal(seed=seed, classification_function=classification_function + 1),
                    seed=seed, position=position, width=width
                )
                if classification_function == 0:
                    n_take = nrows // (num_drifts + 1) + width
                else:
                    n_take = nrows // (num_drifts + 1)
                for x_i, y_i in dataset.take(n_take):
                    data.append((x_i, y_i))
            # Append rest of the stream with classification function where no drift will happen
            dataset = datasets.synth.Agrawal(seed=seed, classification_function=9)
            for x_i, y_i in dataset.take(nrows // (num_drifts + 1) - width):
                data.append((x_i, y_i))
        else:
            if nrows is None: nrows = 10 ** 7
            abrupt_drift, perturbation = False, 0.0
            param = dataset_name[8:]
            if param.startswith('abrupt'):
                abrupt_drift = True
            else:
                perturbation = 0.2
            dataset = datasets.synth.Agrawal(seed=seed, perturbation=perturbation)
            if abrupt_drift:
                data_sliced = []
                for _ in range(num_drifts + 1):
                    for x_i, y_i in dataset.take(nrows // (num_drifts + 1)):
                        data_sliced.append((x_i, y_i))
                    dataset.generate_drift()
                data = data_sliced
            else:
                data = dataset.take(nrows)
        output_logits_dim = 2
        x, y = transform_river_dataset_to_array(data, output_dim=output_logits_dim)

        # One hot encoding of Attributes 3, 4, 5, since they are categorical
        x_one_hot = []
        for val in x:
            row = [val[0], val[1], val[2]]
            row.extend(
                [float(val[3] == 0), float(val[3] == 1), float(val[3] == 2), float(val[3] == 3), float(val[3] == 4)])
            car = [0] * 20
            car[int(val[4]) - 1] = 1
            row.extend(car)
            zipcode = [0] * 9
            zipcode[int(val[5])] = 1
            row.extend(zipcode)
            row.extend(val[6:])
            x_one_hot.append(row)
        x = x_one_hot
    # ----------------------- PMLB SYNTH Data -------------------------
    elif dataset_name.__eq__('sleep'):
        data = pd.read_csv('{}data/generated/seed_{}/oversample_0.{}/sleep.csv'.format(DATA_DIR, seed, oversample_rate),
                           sep=',', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:13].values
        y = data.iloc[:, 13].values
        # Target: {0, 1, 2, 3, 5}
        y = np.array(
            [[float(val == 0), float(val == 1), float(val == 2), float(val == 3), float(val == 5)] for val in y])
        output_logits_dim = 5
    elif dataset_name.__eq__('ann_thyroid'):
        data = pd.read_csv(
            '{}data/generated/seed_{}/oversample_0.{}/ann_thyroid.csv'.format(DATA_DIR, seed, oversample_rate),
            sep=',', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:21].values
        y = data.iloc[:, 21].values
        y = np.array([[float(val == 1), float(val == 2), float(val == 3)] for val in y])
        output_logits_dim = 3
    elif dataset_name.__eq__('spambase'):
        data = pd.read_csv(
            '{}data/generated/seed_{}/oversample_0.{}/spambase.csv'.format(DATA_DIR, seed, oversample_rate),
            sep=',', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:57].values
        y = data.iloc[:, 57].values
        y = np.array([[float(val == 0), float(val == 1)] for val in y])
        output_logits_dim = 2
    elif dataset_name.__eq__('churn'):
        data = pd.read_csv('{}data/generated/seed_{}/oversample_0.{}/churn.csv'.format(DATA_DIR, seed, oversample_rate),
                           sep=',', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:20].values
        y = data.iloc[:, 20].values
        # One hot encoding
        x_one_hot = []
        for row in x:
            row_one_hot = []
            state = [0] * 51
            state[int(row[0])] = 1
            row_one_hot.extend(state)
            row_one_hot.append(row[1])
            row_one_hot.extend([float(row[2] == 408.0), float(row[2] == 510.0), float(row[2] == 415.0)])  # area code
            row_one_hot.extend(row[3:])
            x_one_hot.append(row_one_hot)
        x = x_one_hot
        y = np.array([[float(val == 0), float(val == 1)] for val in y])
        output_logits_dim = 2
    elif dataset_name.__eq__('twonorm'):
        data = pd.read_csv(
            '{}data/generated/seed_{}/oversample_0.{}/twonorm.csv'.format(DATA_DIR, seed, oversample_rate),
            sep=',', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:20].values
        y = data.iloc[:, 20].values
        y = np.array([[float(val == 0), float(val == 1)] for val in y])
        output_logits_dim = 2
    elif dataset_name.__eq__('optdigits'):
        data = pd.read_csv(
            '{}data/generated/seed_{}/oversample_0.{}/optdigits.csv'.format(DATA_DIR, seed, oversample_rate),
            sep=',', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:64].values
        y = data.iloc[:, 64].values
        y = np.array([[float(val == 0), float(val == 1), float(val == 2), float(val == 3), float(val == 4),
                       float(val == 5), float(val == 6), float(val == 7), float(val == 8), float(val == 9)] for val in
                      y])
        output_logits_dim = 10
    elif dataset_name.__eq__('nursery'):
        data = pd.read_csv(
            '{}data/generated/seed_{}/oversample_0.{}/nursery.csv'.format(DATA_DIR, seed, oversample_rate),
            sep=',', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:8].values
        # One hot encoding
        x_encoded = []
        for val in x:
            row = [0] * 26
            row[val[0]] = 1
            row[val[1] + 3] = 1
            row[val[2] + 8] = 1
            row[val[3] + 12] = 1
            row[val[4] + 16] = 1
            row[val[5] + 19] = val[5]
            row[val[6] + 20] = 1
            row[val[7] + 23] = 1
            x_encoded.append(row)
        x = x_encoded
        y = data.iloc[:, 8].values
        y = np.array([[float(val == 0), float(val == 1), float(val == 3), float(val == 4)] for val in y])
        output_logits_dim = 4
        # finance (boolean), all other features are ordinal (they have an ordering), i. e., no nominal attributes
    elif dataset_name.__eq__('texture'):
        data = pd.read_csv(
            '{}data/generated/seed_{}/oversample_0.{}/texture.csv'.format(DATA_DIR, seed, oversample_rate),
            sep=',', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:39].values
        y = data.iloc[:, 40].values
        y = np.array([[float(val == 2), float(val == 3), float(val == 4), float(val == 6), float(val == 7),
                       float(val == 8), float(val == 9), float(val == 10), float(val == 12), float(val == 13),
                       float(val == 14)] for val in y])
        output_logits_dim = 11
    elif dataset_name.__eq__('satimage'):
        data = pd.read_csv(
            '{}data/generated/seed_{}/oversample_0.{}/satimage.csv'.format(DATA_DIR, seed, oversample_rate),
            sep=',', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:36].values
        y = data.iloc[:, 36].values
        y = np.array([[float(val == 1), float(val == 2), float(val == 3), float(val == 4), float(val == 5),
                       float(val == 7)] for val in y])
        output_logits_dim = 6
        # ----------------------- Real World Data -------------------------
    elif dataset_name.__eq__('poker'):
        data = pd.read_csv('{}data/realworld/poker/poker-hand-testing.data'.format(DATA_DIR), sep=',', header=None,
                           nrows=nrows)
        data = data.sample(frac=1, random_state=seed)
        x = data.iloc[:, 0:10].values
        y = data.iloc[:, 10].values
        # One-hot encoding the input
        x_one_hot = []
        for row in x:
            x_one_hot_vec = []
            for i in range(0, 10, 2):
                x_one_hot_vec.extend([int(row[i] == 1), int(row[i] == 2), int(row[i] == 3), int(row[i] == 4)])
                x_one_hot_vec.append(row[i + 1])
            x_one_hot.append(x_one_hot_vec)
        x = x_one_hot
        output_logits_dim = 10
        y = one_hot_encoding_target(y, output_logits_dim)
    elif dataset_name.__eq__('epsilon'):
        x, y = [], []
        with open('{}data/realworld/epsilon_normalized.t'.format(DATA_DIR)) as eps_data:
            for idx, row in enumerate(eps_data):
                label, features = row.split(maxsplit=1)
                x.append(list(map(float, [s.split(':')[1] for s in features.split()])))
                y.append([float(label.__eq__('-1')), float(label.__eq__('1'))])
                if nrows is not None and idx >= nrows: break
        d = list(zip(x, y))
        random.seed(seed)
        random.shuffle(d)
        x, y = zip(*d)
        x, y = list(x), list(y)
        output_logits_dim = 2
    elif dataset_name.__eq__('covertype'):
        data = pd.read_csv('{}data/realworld/Covertype/covtype.data'.format(DATA_DIR), sep=',', header=None,
                           nrows=nrows)
        data = data.sample(frac=1, random_state=seed)
        x = data.iloc[:, 0:54].values
        y = data.iloc[:, 54].values
        y = [val - 1 for val in y]
        # one-hot encoding
        y_np = np.array([[0 for _ in range(1, 8, 1)] for _ in y])
        for i, y_i in enumerate(y):
            y_np[i][y_i - 1] = 1
        y = y_np
        output_logits_dim = 7
    elif dataset_name.__eq__('kdd99'):
        data = pd.read_csv('{}data/realworld/kddcup.data.corrected'.format(DATA_DIR), sep=',', skiprows=0, header=None,
                           nrows=nrows)
        data = data.sample(frac=1, random_state=seed)
        x = data.iloc[:, 0:41].values
        y = data.iloc[:, 41].values

        service = ['sunrpc', 'whois', 'other', 'netstat', 'supdup', 'remote_job', 'http_443', 'finger', 'netbios_ns',
                   'ctf', 'tftp_u', 'rje', 'domain_u', 'eco_i', 'auth', 'efs', 'http_8001', 'netbios_ssn', 'mtp',
                   'http', 'klogin', 'iso_tsap', 'http_2784', 'harvest', 'ldap', 'pop_2', 'kshell', 'netbios_dgm',
                   'ssh', 'red_i', 'nnsp', 'urh_i', 'private', 'hostnames', 'ecr_i', 'daytime', 'name', 'aol', 'tim_i',
                   'sql_net', 'shell', 'uucp_path', 'csnet_ns', 'bgp', 'gopher', 'ftp_data', 'vmnet', 'Z39_50', 'link',
                   'domain', 'time', 'pm_dump', 'imap4', 'exec', 'login', 'telnet', 'pop_3', 'echo', 'ftp', 'systat',
                   'urp_i', 'X11', 'nntp', 'printer', 'ntp_u', 'courier', 'smtp', 'uucp', 'IRC', 'discard']
        flag = ['S1', 'RSTO', 'S2', 'S3', 'S0', 'RSTR', 'SH', 'SF', 'REJ', 'OTH', 'RSTOS0']
        x_np = []
        for x_val in x:
            # duration
            x_one_hot = [x_val[0]]
            # protocol_type
            x_one_hot.extend([int(x_val[1].__eq__('tcp')), int(x_val[1].__eq__('upd')), int(x_val[1].__eq__('icmp'))])
            # service
            tmp = [0] * len(service)
            tmp[service.index(x_val[2])] = 1
            x_one_hot.extend(tmp)
            # flag
            tmp = [0] * len(flag)
            tmp[flag.index(x_val[3])] = 1
            x_one_hot.extend(tmp)
            # rest
            x_one_hot.extend(x_val[4:])
            x_np.append(x_one_hot)
        # ----------- Targets -----------
        targets = {'neptune.': 0, 'ftp_write.': 1, 'warezclient.': 2, 'smurf.': 3, 'portsweep.': 4, 'land.': 5,
                   'spy.': 6,
                   'satan.': 7, 'perl.': 8, 'normal.': 9, 'ipsweep.': 10, 'multihop.': 11, 'warezmaster.': 12,
                   'pod.': 13,
                   'rootkit.': 14, 'phf.': 15, 'guess_passwd.': 16, 'teardrop.': 17, 'loadmodule.': 18, 'nmap.': 19,
                   'buffer_overflow.': 20, 'imap.': 21, 'back.': 22}
        y_np = []
        for y_val in y:
            y_one_hot = [0] * len(targets)
            y_one_hot[targets[y_val]] = 1
            y_np.append(y_one_hot)

        x, y = x_np, y_np
        output_logits_dim = 23

    x = np.array(x)
    y = np.array(y)
    return x, y, output_logits_dim
