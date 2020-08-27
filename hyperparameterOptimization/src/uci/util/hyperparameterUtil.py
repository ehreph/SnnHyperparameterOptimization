import json
import pandas as pd
from pandas import DataFrame

BASE_NAME = "_config"
CONFIG_DIR = './res/configuration/'


def write_hyperparameter_configuration(hp, conf_name):
    """
    saves hyperparemeter configuration to res/configuration
    :param hp: hyperparameters as dict
    :param conf_name: string

    """
    file_name = conf_name + BASE_NAME + '.json'
    with open(CONFIG_DIR + '{0}'.format(file_name), 'w') as json_file:
        json.dump(hp, json_file, indent=4)


def load_hyperparameter_configuration(conf_name):
    """
    loads hyperparemeter configuration from res/configuration

    :param conf_name:
    :return: hp as dict
    """
    file_name = conf_name + BASE_NAME + '.json'
    with open(CONFIG_DIR + '{0}'.format(file_name), 'r') as json_file:
        conf = json.loads(json_file.read())
    return conf


def load_hyperparameter_set(file_name):
    """
    load hyperparameter_configuration_set from res/configuration

    :param file_name:
    :return:
    """
    hp_set = pd.read_csv(CONFIG_DIR + file_name + BASE_NAME + '.csv').dropna()
    return hp_set


def convert_to_dict(hp_set: DataFrame):
    """
    convert pandas Dataframe to hyperparameter dictionary

    :param hp_set:
    :return hp_dict:
    """
    dataset_dict = hp_set.set_index('dataset').T
    hp_dict = dataset_dict.to_dict("dict")
    return hp_dict


def get_dataset_dict(hp_dict, name):
    hp = hp_dict.get(name)
    if hp is not None:
        del hp['valAccuracy']
        del hp['testAccuracy']
    return hp
