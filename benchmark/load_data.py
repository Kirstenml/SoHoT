from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from river import datasets
import random

DATA_DIR = ""


def get_data_loader(dataset_name, batch_size=32, nrows=None, drop_last=True, return_data=False,
                    oversample_rate=75, seed=42):
    # ----------------------- Synthetic Streams -----------------------
    if dataset_name.startswith('sea'):
        if dataset_name[3:].startswith('50'):
            noise = 50
        else:
            noise = 5 * (10 ** 5)
        data = Sea(nrows=nrows, noise=noise, seed=seed)
        output_dim = 2
        input_dim = 3
    elif dataset_name.startswith('randomrbf'):
        a, b = dataset_name[9:].split('_')
        n_drift_centroids = int(a)
        change_speed = float('0.' + b)
        data = RandomRBFDrift(nrows=nrows, seed_model=seed, seed_sample=seed, change_speed=change_speed,
                              n_drift_centroids=n_drift_centroids)
        output_dim = 5
        input_dim = 10
    elif dataset_name.startswith('hyperplane'):
        a, b = dataset_name[10:].split('_')
        n_drift_features = int(a)
        mag_change = float('0.' + b)
        data = Hyperplane(nrows=nrows, seed=seed, n_drift_features=n_drift_features, mag_change=mag_change)
        output_dim = 2
        input_dim = 10
    elif dataset_name.startswith('agrawal'):
        abrupt_drift, random_abrupt_drift, perturbation = False, False, 0.0
        param = dataset_name[8:]
        if param.startswith('abrupt'):
            abrupt_drift = True
        elif param.startswith('random_abrupt'):
            random_abrupt_drift = True
        else:
            perturbation = 0.2
        if nrows is None: nrows = 10 ** 7
        data = Agrawal(nrows=nrows, seed=seed, abrupt_drift=abrupt_drift, perturbation=perturbation,
                       random_abrupt_drift=random_abrupt_drift)
        input_dim = 40
        output_dim = 2
    # ----------------------- PMLB SYNTH Data -----------------------
    elif dataset_name.__eq__('sleep'):
        data = Sleep(data_dir=DATA_DIR, nrows=nrows, oversample_rate=oversample_rate, seed=seed)
        input_dim = 21  # before one hot encoding 13
        output_dim = 5
    elif dataset_name.__eq__('ann_thyroid'):
        data = Ann_thyroid(data_dir=DATA_DIR, nrows=nrows, oversample_rate=oversample_rate, seed=seed)
        input_dim = 21
        output_dim = 3
    elif dataset_name.__eq__('spambase'):
        data = Spambase(data_dir=DATA_DIR, nrows=nrows, oversample_rate=oversample_rate, seed=seed)
        input_dim = 57
        output_dim = 2
    elif dataset_name.__eq__('churn'):
        data = Churn(data_dir=DATA_DIR, nrows=nrows, oversample_rate=oversample_rate, seed=seed)
        input_dim = 72
        output_dim = 2
    elif dataset_name.startswith('nursery'):
        data = Nursery(data_dir=DATA_DIR, nrows=nrows, oversample_rate=oversample_rate, seed=seed)
        input_dim = 26
        output_dim = 4
    elif dataset_name.startswith('twonorm'):
        data = Twonorm(data_dir=DATA_DIR, nrows=nrows, oversample_rate=oversample_rate, seed=seed)
        input_dim = 20
        output_dim = 2
    elif dataset_name.__eq__('optdigits'):
        data = Optdigits(data_dir=DATA_DIR, nrows=nrows, oversample_rate=oversample_rate, seed=seed)
        input_dim = 64
        output_dim = 10
    elif dataset_name.__eq__('texture'):
        data = Texture(data_dir=DATA_DIR, nrows=nrows, oversample_rate=oversample_rate, seed=seed)
        input_dim = 40
        output_dim = 11
    elif dataset_name.__eq__('satimage'):
        data = Satimage(data_dir=DATA_DIR, nrows=nrows, oversample_rate=oversample_rate, seed=seed)
        input_dim = 36
        output_dim = 6
    # ----------------------- Real World Data -----------------------
    elif dataset_name.__eq__('poker'):
        data = PokerHand(data_dir=DATA_DIR, nrows=nrows, seed=seed)
        input_dim = 25
        output_dim = 10
    elif dataset_name.startswith('covertype'):
        data = Covertype(data_dir=DATA_DIR, nrows=nrows, seed=seed)
        input_dim = 54
        output_dim = 7
    elif dataset_name.__eq__('kdd99'):
        data = Kdd99(data_dir=DATA_DIR, nrows=nrows, seed=seed)
        input_dim = 122
        output_dim = 23
    elif dataset_name.__eq__('epsilon'):
        data = Epsilon(data_dir=DATA_DIR, nrows=nrows, seed=seed)
        input_dim = 2000
        output_dim = 2

    if return_data:
        return data, input_dim, output_dim
    loader = DataLoader(data, batch_size=batch_size, drop_last=drop_last)
    return loader, input_dim, output_dim


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- SYNTHETIC STREAMS ----------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# MOA/River streams generator
def transform_river_dataset_to_torch(data, encoding_fn=None):
    x, y = [], []
    for i, (x_i, y_i) in enumerate(data):
        x_vec = list(x_i.values())
        if encoding_fn is not None: x_vec = encoding_fn(x_vec)
        # Label integer encoding
        y.append(int(y_i))
        x.append(x_vec)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class Sea(Dataset):
    def __init__(self, nrows=None, noise=50, seed=42):
        if nrows is None: nrows = 10 ** 6
        dataset = datasets.synth.SEA(noise=noise, seed=seed)
        self.X_train, self.y_train = transform_river_dataset_to_torch(dataset.take(nrows))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class Hyperplane(Dataset):
    def __init__(self, nrows=None, seed=42, n_drift_features=10, mag_change=0.001):
        if nrows is None: nrows = 10 ** 7
        dataset = datasets.synth.Hyperplane(seed=seed, n_features=10, n_drift_features=n_drift_features,
                                            mag_change=mag_change)
        # Elapsed time 38.46 seconds
        self.X_train, self.y_train = transform_river_dataset_to_torch(dataset.take(nrows))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class RandomRBFDrift(Dataset):
    def __init__(self, nrows=None, seed_model=42, seed_sample=42, change_speed=0.001, n_drift_centroids=50):
        if nrows is None: nrows = 10 ** 7
        dataset = datasets.synth.RandomRBFDrift(seed_model=seed_model, seed_sample=seed_sample,
                                                n_features=10, n_classes=5, n_centroids=50,
                                                change_speed=change_speed,
                                                n_drift_centroids=n_drift_centroids)
        self.X_train, self.y_train = transform_river_dataset_to_torch(dataset.take(nrows))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class Agrawal(Dataset):
    def __init__(self, nrows=None, seed=42, abrupt_drift=False, perturbation=0.0, num_drifts=9,
                 random_abrupt_drift=False):
        dataset = datasets.synth.Agrawal(seed=seed, perturbation=perturbation)
        if abrupt_drift:
            data_sliced = []
            for _ in range(num_drifts + 1):
                # Iterate over the dataset to be able to call generate_drift (otherwise self._rng is not initialized)
                for x_i, y_i in dataset.take(nrows // (num_drifts + 1)):
                    data_sliced.append((x_i, y_i))
                dataset.generate_drift()
            data = data_sliced
        elif random_abrupt_drift:
            drift_positions = [random.randint(0, nrows) for _ in range(num_drifts + 1)]
            drift_positions.sort()
            drift_positions.append(nrows)
            data_sliced = []
            for idx in range(num_drifts + 1):
                for x_i, y_i in dataset.take(drift_positions[idx + 1] - drift_positions[idx]):
                    data_sliced.append((x_i, y_i))
                dataset.generate_drift()
            data = data_sliced
        else:
            data = dataset.take(nrows)
        self.X_train, self.y_train = transform_river_dataset_to_torch(data, encoding_fn=Agrawal.one_hot_encoding)

    @staticmethod
    def one_hot_encoding(val):
        # Attributes 3, 4, 5 are categorical
        row = [val[0], val[1], val[2]]
        row.extend([float(val[3] == 0), float(val[3] == 1), float(val[3] == 2), float(val[3] == 3), float(val[3] == 4)])
        car = [0] * 20
        car[int(val[4]) - 1] = 1
        row.extend(car)
        zipcode = [0] * 9
        zipcode[int(val[5])] = 1
        row.extend(zipcode)
        row.extend(val[6:])
        return row

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

# ----------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- SYNTH PMLB STREAMS ----------------------------------------------
# -------- Original data is from: https://epistasislab.github.io/pmlb/ -------------------------------------------
# -------- Synthetically generate a drifting data stream using CTGAN  --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# Feature V3 is categorical, values: {0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, rest is continuous
class Sleep(Dataset):
    def __init__(self, data_dir="", nrows=None, synthetic_generated=True, oversample_rate=75, seed=42):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/sleep.csv'.format(data_dir, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/sleep.tsv'.format(data_dir), sep='\t', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:13].values
        y = data.iloc[:, 13].values
        # one-hot encoding
        v3_one_hot = {0.0: 0, 2.0: 1, 4.0: 2, 5.0: 3, 6.0: 4, 7.0: 5, 8.0: 6, 9.0: 7, 10.0: 8}
        x_one_hot = []
        for row in x:
            row_one_hot = row[:3].tolist()
            v3 = [0] * 9
            v3[v3_one_hot[row[3]]] = 1
            row_one_hot.extend(v3)
            row_one_hot.extend(row[4:])
            x_one_hot.append(row_one_hot)
        y = np.array([y_val - 1 if y_val == 5 else y_val for y_val in y])

        self.X_train = torch.tensor(x_one_hot, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# Features: 15 binary (A2-A16)
class Ann_thyroid(Dataset):
    def __init__(self, data_dir="", nrows=None, synthetic_generated=True, oversample_rate=75, seed=42):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/ann_thyroid.csv'.format(data_dir, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/ann-thyroid.tsv'.format(data_dir), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:21].values
        y = data.iloc[:, 21].values
        y = np.array([val - 1 for val in y])

        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# All features are continuous
class Spambase(Dataset):
    def __init__(self, data_dir="", nrows=None, synthetic_generated=True, oversample_rate=75, seed=42):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/spambase.csv'.format(data_dir, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)

        else:
            data = pd.read_csv('{}data/pmlb/spambase.tsv'.format(data_dir), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:57].values
        y = data.iloc[:, 57].values
        y = np.array(y)
        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# Features: 2 categorical (state, area code), 2 binary (international plan, voice mail plan), 16 continuous
class Churn(Dataset):
    def __init__(self, data_dir="", nrows=None, synthetic_generated=True, oversample_rate=75, seed=42):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/churn.csv'.format(data_dir, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/churn.tsv'.format(data_dir), sep='\t', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:20].values
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
        y = data.iloc[:, 20].values
        y = np.array(y)
        self.X_train = torch.tensor(x_one_hot, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# All features are real valued
class Twonorm(Dataset):
    def __init__(self, data_dir="", nrows=None, synthetic_generated=True, oversample_rate=75, seed=42):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/twonorm.csv'.format(data_dir, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)

        else:
            data = pd.read_csv('{}data/pmlb/twonorm.tsv'.format(data_dir), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:20].values
        y = data.iloc[:, 20].values
        y = np.array(y)
        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# Input57 is Boolean
class Optdigits(Dataset):
    def __init__(self, data_dir="", nrows=None, synthetic_generated=True, oversample_rate=75, seed=42):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/optdigits.csv'.format(data_dir, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)

        else:
            data = pd.read_csv('{}data/pmlb/optdigits.tsv'.format(data_dir), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:64].values
        y = data.iloc[:, 64].values
        y = np.array(y)
        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# Goal: rank applications for nursery schools.
# Attributes and their values
# parents {0, 1, 2}:  usual, pretentious, great_pret
# has_nurs {0, 1, 2, 3, 4}:  proper, less_proper, improper, critical, very_crit
# form {0, 1, 2, 3}: complete, completed, incomplete, foster
# children {0, 1, 2, 3}:  1, 2, 3, more
# housing {0, 1, 2}: convenient, less_conv, critical
# finance {0, 1}: convenient, inconv -> Binary, no one-hot encoding necessary
# social {0, 1, 2}: non-prob, slightly_prob, problematic
# health {0, 1, 2}: recommended, priority, not_recom
class Nursery(Dataset):
    def __init__(self, data_dir="", nrows=None, synthetic_generated=True, oversample_rate=75, seed=42):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/nursery.csv'.format(data_dir, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/nursery.tsv'.format(data_dir), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
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
        y = [val - 1 if val > 2 else val for val in y]
        y = np.array(y)
        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class Texture(Dataset):
    def __init__(self, data_dir="", nrows=None, synthetic_generated=True, oversample_rate=75, seed=42):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/texture.csv'.format(data_dir, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)

        else:
            data = pd.read_csv('{}data/pmlb/texture.tsv'.format(data_dir), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:40].values
        y = data.iloc[:, 40].values
        targets = {2: 0, 3: 1, 4: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 12: 8, 13: 9, 14: 10}
        y = np.array([targets[val] for val in y])
        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# All features are continuous
class Satimage(Dataset):
    def __init__(self, data_dir="", nrows=None, synthetic_generated=True, oversample_rate=75, seed=42):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/satimage.csv'.format(data_dir, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)

        else:
            data = pd.read_csv('{}data/pmlb/satimage.tsv'.format(data_dir), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:36].values
        y = data.iloc[:, 36].values
        targets = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5}
        y = np.array([targets[val] for val in y])
        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# ----------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- REAL WORLD STREAMS ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


# UCI Poker hand repository
# source: https://archive.ics.uci.edu/ml/datasets/Poker+Hand
class PokerHand(Dataset):
    def __init__(self, data_dir="", nrows=None, seed=42):
        data = pd.read_csv('{}data/realworld/poker/poker-hand-testing.data'.format(data_dir), sep=',', header=None)
        data2 = pd.read_csv('{}data/realworld/poker/poker-hand-training-true.data'.format(data_dir), sep=',',
                            header=None)
        data = pd.concat([data, data2])
        # shuffle the data with random seed
        data = data.sample(frac=1, random_state=seed)
        x_inp = data.iloc[:, 0:10].values
        # One-hot encoding the input
        x = []
        for row in x_inp:
            x_one_hot = []
            for i in range(0, 10, 2):
                x_one_hot.extend([int(row[i] == 1), int(row[i] == 2), int(row[i] == 3), int(row[i] == 4)])
                x_one_hot.append(row[i + 1])
            x.append(x_one_hot)
        y_numbers = data.iloc[:, 10].values
        # since it is a classification task, one-hot encoded
        y = np.array(y_numbers)

        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# source: https://archive.ics.uci.edu/dataset/31/covertype
class Covertype(Dataset):
    def __init__(self, data_dir="", nrows=None, seed=42):
        data = pd.read_csv('{}data/realworld/Covertype/covtype.data'.format(data_dir), sep=',', header=None,
                           nrows=nrows)
        # shuffle the data with random seed
        data = data.sample(frac=1, random_state=seed)
        x = data.iloc[:, 0:54].values
        y = data.iloc[:, 54].values
        y = np.array([val - 1 for val in y])

        self.X_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)
        self.attribute_list = {0: 'Elevation', 1: 'Aspect', 2: 'Slope', 3: 'Horizontal_Distance_To_Hydrology',
                               4: 'Vertical_Distance_To_Hydrology', 5: 'Horizontal_Distance_To_Roadways',
                               6: 'Hillshade_9am', 7: 'Hillshade_Noon', 8: 'Hillshade_3pm',
                               9: 'Horizontal_Distance_To_Fire_Points', 10: 'Wilderness_Area_1',
                               11: 'Wilderness_Area_2',
                               12: 'Wilderness_Area_3', 13: 'Wilderness_Area_4'}
        for i in range(40): self.attribute_list[14 + i] = 'Soil_Type_{}'.format(i)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def get_attribute_list(self):
        return self.attribute_list


# Popular dataset (see https://kdd.ics.uci.edu/databases/kddcup99/task.html), used in e. g.
#   N. Gunasekara, H. M. Gomes, B. Pfahringer and A. Bifet, "Online Hyperparameter Optimization for Streaming Neural
#   Networks," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-9
class Kdd99(Dataset):
    def __init__(self, data_dir="", nrows=None, seed=42):
        data = pd.read_csv('{}data/realworld/kddcup.data.corrected'.format(data_dir), sep=',', skiprows=0, header=None,
                           nrows=nrows)
        # shuffle the data with random seed
        data = data.sample(frac=1, random_state=seed)
        x = data.iloc[:, 0:41].values
        y = data.iloc[:, 41].values
        # ----------- Features - one hot -----------
        # one-hot encoding: 1. protocol_type (tcp,upd,icmp), 2. service (http,telnet, ...) [70 in total],
        # 3. flag ('S1', 'RSTO', 'S2', 'S3', 'S0', 'RSTR', 'SH', 'SF', 'REJ', 'OTH', 'RSTOS0'),
        # 6. land (0,1), 11. logged_in (0,1), root_shell (0,1) [not listed], su_attempted (0,1) [not listed],
        # 20. is_hot_login (0,1), 21. is_guest_login (0,1)
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
            # remaining features
            x_one_hot.extend(x_val[4:])
            x_np.append(x_one_hot)
        # ----------- Targets -----------
        targets = {'neptune.': 0, 'ftp_write.': 1, 'warezclient.': 2, 'smurf.': 3, 'portsweep.': 4, 'land.': 5,
                   'spy.': 6,
                   'satan.': 7, 'perl.': 8, 'normal.': 9, 'ipsweep.': 10, 'multihop.': 11, 'warezmaster.': 12,
                   'pod.': 13,
                   'rootkit.': 14, 'phf.': 15, 'guess_passwd.': 16, 'teardrop.': 17, 'loadmodule.': 18, 'nmap.': 19,
                   'buffer_overflow.': 20, 'imap.': 21, 'back.': 22}
        y = [targets[val] for val in y]
        self.X_train = torch.tensor(np.array(x_np), dtype=torch.float32)
        self.y_train = torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


# High dimensional dataset
# used in N. Gunasekara et al.: "Online Hyperparameter Optimization for Streaming Neural Networks"
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
class Epsilon(Dataset):
    def __init__(self, data_dir="", nrows=None, seed=42):
        x, y = [], []
        with open('{}data/realworld/epsilon_normalized.t'.format(data_dir)) as eps_data:
            for idx, row in enumerate(eps_data):
                label, features = row.split(maxsplit=1)
                x.append(list(map(float, [s.split(':')[1] for s in features.split()])))
                y.append(0 if label.__eq__('-1') else 1)
                if nrows is not None and idx >= nrows: break
        d = list(zip(x, y))
        random.seed(seed)
        random.shuffle(d)
        x, y = zip(*d)
        self.X_train = torch.tensor(np.array(x), dtype=torch.float32)
        self.y_train = torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
