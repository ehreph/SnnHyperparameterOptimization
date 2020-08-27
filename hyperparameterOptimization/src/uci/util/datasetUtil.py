import pandas as pd

import src.uci.entity.datasetConfig as datasetConfig
from src.uci.entity.datasetConfig import DatasetConfig
from src.uci.util.util import get_dataset_file_name, get_dataset_label_file_name

RESOURCE_DIR = './res/'
CONFIG_DIR = RESOURCE_DIR + 'configuration/'


def load_datasets_config(file_name):
    '''
    load datasets-configurations from configuration directory

    :param file_name:
    :return:
    '''
    datasets_config = pd.read_csv(CONFIG_DIR + file_name + '.csv').dropna()
    array = datasets_config.to_numpy()
    datasets = []
    for row in array:
        conf = datasetConfig.DatasetConfig(name=row[0],
                                           folder=row[1],
                                           file_name=row[2],
                                           task=row[3],
                                           prediction_column=row[4])
        datasets.append(conf)

    return datasets


def load_dataset(dataset: DatasetConfig, delimiter=','):
    '''
    loads dataset and returns numpy array with data

    :param dataset:
    :param delimiter:
    :return: numpy.darray
    '''
    # load data of dataset
    path = get_dataset_file_name(dataset)
    label_path = get_dataset_label_file_name(dataset)
    x = pd.read_csv(path, header=None, delimiter=delimiter).to_numpy()
    y = pd.read_csv(label_path, header=None, delimiter=delimiter).to_numpy()

    return x, y
