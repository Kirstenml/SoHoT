import random
import pandas as pd
from river import datasets

# TODO Set directory to data
DATA_DIR = ""

def get_output_dims(data):
    output_dims = {'sea50': 2, 'sea5E5': 2, 'hyperplane10_0001': 2, 'hyperplane10_001': 2,
                   'randomrbf50_0001': 5, 'randomrbf50_001': 5,
                   'agrawal_abrupt_drift': 2, 'agrawal_perturbation': 2,
                   'sleep': 5, 'nursery': 4, 'twonorm': 2, 'ann_thyroid': 3, 'satimage': 6, 'optdigits': 10,
                   'churn': 2, 'texture': 11, 'spambase': 2,
                   'poker': 10, 'kdd99': 23, 'covertype': 7, 'epsilon': 2
                   }
    return output_dims[data]


# HT expects a dict as input
def river_input_format(x):
    x_transformed = []
    for x_i in x:
        x_transformed.append(dict(zip([i_i for i_i in range(len(x))], x_i)))
    return x_transformed


def load_data_ht(dataset_name, nrows=None, synthetic_generated=True, oversample_rate=75, distilbert=False, seed=42):
    nominal_attributes = []
    # ----------------------- Synthetic Streams -----------------------
    if dataset_name.startswith('sea'):
        if nrows is None: nrows = 10 ** 6
        if dataset_name[3:].__eq__('50'):
            noise = 50
        else:
            noise = 5 * (10 ** 5)
        dataset = datasets.synth.SEA(noise=noise, seed=seed)
        data = dataset.take(nrows)
    elif dataset_name.startswith('hyperplane'):
        if nrows is None: nrows = 10 ** 7
        a, b = dataset_name[10:].split('_')
        n_drift_features = int(a)
        mag_change = float('0.' + b)
        dataset = datasets.synth.Hyperplane(seed=seed, n_features=10, n_drift_features=n_drift_features,
                                            mag_change=mag_change)
        data = dataset.take(nrows)
    elif dataset_name.startswith('randomrbf'):
        if nrows is None: nrows = 10 ** 7
        a, b = dataset_name[9:].split('_')
        n_drift_centroids = int(a)
        change_speed = float('0.' + b)
        dataset = datasets.synth.RandomRBFDrift(seed_model=seed, seed_sample=seed,
                                                n_features=10, n_classes=5, n_centroids=50,
                                                change_speed=change_speed,
                                                n_drift_centroids=n_drift_centroids)
        data = dataset.take(nrows)
    elif dataset_name.startswith('agrawal'):
        if nrows is None: nrows = 10 ** 7
        num_drifts = 10
        abrupt_drift, random_abrupt_drift, perturbation = False, False, 0.0
        param = dataset_name[8:]
        if param.startswith('abrupt'):
            abrupt_drift = True
        elif param.startswith('random_abrupt'):
            random_abrupt_drift = True
        else:
            perturbation = 0.2
        dataset = datasets.synth.Agrawal(seed=seed, perturbation=perturbation)
        if abrupt_drift:
            data_sliced = []
            for _ in range(num_drifts):
                # Iterate over the dataset to be able to call generate_drift (otherwise self._rng is not initialized)
                for x_i, y_i in dataset.take(nrows // num_drifts):
                    data_sliced.append((x_i, y_i))
                dataset.generate_drift()
            data = data_sliced
        elif random_abrupt_drift:
            drift_positions = [random.randint(0, nrows) for _ in range(num_drifts)]
            drift_positions.sort()
            drift_positions.append(nrows)
            data_sliced = []
            for idx in range(num_drifts):
                # Iterate over the dataset to be able to call generate_drift (otherwise self._rng is not initialized)
                for x_i, y_i in dataset.take(drift_positions[idx + 1] - drift_positions[idx]):
                    data_sliced.append((x_i, y_i))
                dataset.generate_drift()
            data = data_sliced
        else:
            data = dataset.take(nrows)
        nominal_attributes = [3, 4, 5]
    # ----------------------- PMLB SYNTH Data -------------------------
    elif dataset_name.__eq__('sleep'):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/sleep.csv'.format(DATA_DIR, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/sleep.tsv'.format(DATA_DIR), sep='\t', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:13].values
        y = data.iloc[:, 13].values
        # Target: {0, 1, 2, 3, 5}
        y = [y_val - 1 if y_val == 5 else y_val for y_val in y]
        data = zip(river_input_format(x), y)
        nominal_attributes = [3]
    elif dataset_name.__eq__('ann_thyroid'):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/ann_thyroid.csv'.format(DATA_DIR, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/ann-thyroid.tsv'.format(DATA_DIR), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:21].values
        y = data.iloc[:, 21].values
        y = [val - 1 for val in y]
        data = zip(river_input_format(x), y)
    elif dataset_name.__eq__('spambase'):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/spambase.csv'.format(DATA_DIR, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/spambase.tsv'.format(DATA_DIR), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:57].values
        y = data.iloc[:, 57].values
        data = zip(river_input_format(x), y)
    elif dataset_name.__eq__('churn'):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/churn.csv'.format(DATA_DIR, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/churn.tsv'.format(DATA_DIR), sep='\t', skiprows=1, header=None, nrows=nrows)
        x = data.iloc[:, 0:20].values
        y = data.iloc[:, 20].values
        data = zip(river_input_format(x), y)
        nominal_attributes = [0, 2]
    elif dataset_name.__eq__('nursery'):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/nursery.csv'.format(DATA_DIR, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/nursery.tsv'.format(DATA_DIR), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:8].values
        y = data.iloc[:, 8].values
        y = [val - 1 if val > 2 else val for val in y]
        data = zip(river_input_format(x), y)
        # finance (boolean), all other features are ordinal (they have an ordering), i. e., no nominal attributes
    elif dataset_name.startswith('twonorm'):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/twonorm.csv'.format(DATA_DIR, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/twonorm.tsv'.format(DATA_DIR), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:20].values
        y = data.iloc[:, 20].values
        data = zip(river_input_format(x), y)
    elif dataset_name.__eq__('optdigits'):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/optdigits.csv'.format(DATA_DIR, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/optdigits.tsv'.format(DATA_DIR), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:64].values
        y = data.iloc[:, 64].values
        data = zip(river_input_format(x), y)
    elif dataset_name.__eq__('texture'):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/texture.csv'.format(DATA_DIR, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/texture.tsv'.format(DATA_DIR), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:39].values
        y = data.iloc[:, 40].values
        targets = {2: 0, 3: 1, 4: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 12: 8, 13: 9, 14: 10}
        y = [targets[val] for val in y]
        data = zip(river_input_format(x), y)
    elif dataset_name.__eq__('satimage'):
        if synthetic_generated:
            data = pd.read_csv(
                '{}data/generated/seed_{}/oversample_0.{}/satimage.csv'.format(DATA_DIR, seed, oversample_rate),
                sep=',', skiprows=1, header=None, nrows=nrows)
        else:
            data = pd.read_csv('{}data/pmlb/satimage.tsv'.format(DATA_DIR), sep='\t', skiprows=1, header=None,
                               nrows=nrows)
        x = data.iloc[:, 0:36].values
        y = data.iloc[:, 36].values
        targets = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5}
        y = [targets[val] for val in y]
        data = zip(river_input_format(x), y)
    # ----------------------- Real World Data -------------------------
    elif dataset_name.__eq__('poker'):
        data = pd.read_csv('{}data/realworld/poker/poker-hand-testing.data'.format(DATA_DIR), sep=',', header=None,
                           nrows=nrows)
        data = data.sample(frac=1, random_state=seed)
        x = data.iloc[:, 0:10].values
        y = data.iloc[:, 10].values
        data = zip(river_input_format(x), y)
        nominal_attributes = [0, 2, 4, 6, 8]
    elif dataset_name.__eq__('epsilon'):
        x, y = [], []
        with open('{}data/realworld/epsilon_normalized.t'.format(DATA_DIR)) as eps_data:
            for idx, row in enumerate(eps_data):
                label, features = row.split(maxsplit=1)
                x.append(list(map(float, [s.split(':')[1] for s in features.split()])))
                y.append(0 if label.__eq__('-1') else 1)
                if nrows is not None and idx >= nrows: break
        d = list(zip(x, y))
        random.seed(seed)
        random.shuffle(d)
        x, y = zip(*d)
        x, y = list(x), list(y)
        data = zip(river_input_format(x), y)
    elif dataset_name.startswith('covertype'):
        data = pd.read_csv('{}data/realworld/Covertype/covtype.data'.format(DATA_DIR), sep=',', header=None,
                           nrows=nrows)
        data = data.sample(frac=1, random_state=seed)
        x = data.iloc[:, 0:54].values
        y = data.iloc[:, 54].values
        y = [val - 1 for val in y]
        data = zip(river_input_format(x), y)
    elif dataset_name.__eq__('kdd99'):
        data = pd.read_csv('{}data/realworld/kddcup.data.corrected'.format(DATA_DIR), sep=',', skiprows=0, header=None,
                           nrows=nrows)
        data = data.sample(frac=1, random_state=seed)
        x = data.iloc[:, 0:41].values
        y = data.iloc[:, 41].values
        targets = {'neptune.': 0, 'ftp_write.': 1, 'warezclient.': 2, 'smurf.': 3, 'portsweep.': 4, 'land.': 5,
                   'spy.': 6, 'satan.': 7, 'perl.': 8, 'normal.': 9, 'ipsweep.': 10, 'multihop.': 11,
                   'warezmaster.': 12, 'pod.': 13, 'rootkit.': 14, 'phf.': 15, 'guess_passwd.': 16, 'teardrop.': 17,
                   'loadmodule.': 18, 'nmap.': 19, 'buffer_overflow.': 20, 'imap.': 21, 'back.': 22}
        y = [targets[val] for val in y]
        # Reformatting of the data to River format takes ages, it is best to perform the zipping only during evaluation
        data = zip(river_input_format(x), y)
        nominal_attributes = [1, 2, 3]
    return data, nominal_attributes
