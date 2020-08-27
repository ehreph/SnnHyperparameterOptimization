from keras.models import load_model

import pandas as pd
from src.uci.entity.datasetConfig import DatasetConfig
import os

os.chdir('..')
os.chdir('..')
TARGET_DIR = 'target/'
TIME_FILE = 'runtime_measure.csv'
RESOURCE_DIR = 'res/'
BASE_DIR = RESOURCE_DIR + 'dataset/'
MODEL_DIR = 'compiledModels/'
RESULT_DIR = 'results/'


def save_network(model, dataset, model_name):
    file_path = get_file_path(dataset, model_name)
    print("save model at: {0}".format(file_path))
    model.save(file_path)


def load_network(dataset, model_name):
    file_path = get_file_path(dataset, model_name)
    try:
        print("load model at: {0}".format(file_path))
        loaded_model = load_model(file_path)
        return loaded_model
    except ImportError:
        raise
    except OSError:
        print("file could not be found at path: {0}".format(file_path))
        raise ImportError


def get_file_path(dataset: DatasetConfig, name):
    directory = "./" + TARGET_DIR + dataset.folder + "/"
    model_dir = directory + MODEL_DIR

    create_not_available_dir(directory)
    create_not_available_dir(model_dir)
    return model_dir + name + ".h5"


def get_dataset_path(dataset: DatasetConfig):
    return "./" + BASE_DIR + dataset.folder + "/"


def get_dataset_file_name(dataset: DatasetConfig):
    path = get_dataset_path(dataset)
    return path + dataset.file_name + '_py.dat'


def get_dataset_label_file_name(dataset: DatasetConfig):
    path = get_dataset_path(dataset)
    return path + 'labels_py.dat'


def get_next_layer_size(n_out, n_in, n_dense, neurons):
    '''
    formular :  a = (n_out / n_in) ** (1 / n_dense)
    the method returns the number of neurons
    according to the geometric progression
    the result is rounded

    :param n_out:
    :param n_in
    :param n_dense:
    :param neurons:
    :return: conic_factor
    '''
    conic_factor = (n_out / n_in) ** (1 / n_dense)
    size = round(neurons * conic_factor)
    return size if size > n_out else n_out


def load_network_set(dataset: DatasetConfig, base_model_name="snn", amount=1):
    try:
        for i in range(amount):
            model_name = base_model_name + str(i)
            l_model = load_network(dataset, model_name)
            if l_model:
                yield l_model
    except ImportError:
        print("could not load all models correctly")
    return


def create_results_table(data):
    columns = list(
        ["dataset", "rows", "features", "task", "model_name", "test_score", "test_accuracy", "hyperparameters"])
    result = pd.DataFrame(data=data[0:, 0:],  # values
                          columns=columns)
    return result


def save_dataset_results(dataset, result):
    res_dir = get_resource_dir()
    result.to_csv(res_dir + "{0}_result.csv".format(dataset.file_name), index_label=False, index=False)


def get_resource_dir():
    res_dir = TARGET_DIR + RESULT_DIR
    create_not_available_dir(res_dir)
    return res_dir


def create_not_available_dir(directory):
    import os
    if not os.path.isdir(directory):
        os.mkdir(directory)


def save_time_measure(dataset_name, start_time, stop_time):
    time_measure = pd.DataFrame()
    row = dict()
    row['dataset'] = dataset_name
    row['start_time'] = start_time
    row['stop_time'] = stop_time
    row['time'] = stop_time - start_time
    time_measure = time_measure.append(row, ignore_index=True)

    if not os.path.isfile(TARGET_DIR + TIME_FILE):
        time_measure.to_csv(TARGET_DIR + TIME_FILE, index=False)
    else:  # else it exists so append without writing the header
        time_measure.to_csv(TARGET_DIR + TIME_FILE, mode='a', header=False, index=False)


# create target director if not existent
create_not_available_dir(TARGET_DIR)
