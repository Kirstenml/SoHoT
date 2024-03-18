import pandas as pd


def write_loss_to_file(losses, accuracy, auroc, ce_loss_avg=0, dataset_name='', seed=0, extension_at_batch=[]):
    if dataset_name.startswith('sohot'):
        algorithm = 'sohot'
    elif dataset_name.startswith('ht'):
        algorithm = 'ht'
    else:
        algorithm = 'tel'
    sample_idx = range(len(losses))
    if extension_at_batch:
        df = pd.DataFrame({'Extension on batch' : extension_at_batch})
        f = open('evaluation/losses/{}_extension_seed_{}.csv'.format(dataset_name, seed), "w")
        f.write("# Growth of the SoHoT Ensemble at a batch:\n")
        df.to_csv(f, index=False, lineterminator='\n')
        f.close()

    # columns: Cross Entropy Loss using class probabilities, Fading losses
    df = pd.DataFrame({'Sample': sample_idx,
                       'CE Loss {}'.format(algorithm): losses})
    f = open('evaluation/losses/{}_losses_seed_{}.csv'.format(dataset_name, seed), "w")
    f.write('# {} accuracy: {}, roc auc: {}, ce loss avg: {} \n'.format(algorithm, accuracy, auroc, ce_loss_avg))
    df.to_csv(f, index=False, lineterminator='\n')
    f.close()
