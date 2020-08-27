import pandas as pd
import keras as K
import time as time
import tensorflow as tf
from src.uci.network import snn
from src.uci.util.datasetUtil import load_datasets_config

from src.uci.util.preprocessUtil import preprocess_dataset
from src.uci.util.hyperparameterUtil import load_hyperparameter_set, convert_to_dict, get_dataset_dict
from tensorflow.python.client import device_lib
from src.uci.util.util import save_dataset_results, save_time_measure

print("Keras version: " + K.__version__)
# Configure Tensorflow and Keras to run with GPU

print(device_lib.list_local_devices())

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 16})
config.gpu_options.allow_growth = True

data_config = load_datasets_config('datasets_config_complete')
hp_set = load_hyperparameter_set('snn_hyperparameter')
hp_dict = convert_to_dict(hp_set)
load_models = False
save_models = False


def create_new_tf_session():
    K.backend.clear_session()
    s = tf.Session(config=config)
    K.backend.set_session(s)


# create initial KERAS session
create_new_tf_session()

for dataset in data_config:
    start_time = time.time()
    print("start analyzing dataset {0}".format(dataset.name))

    data = preprocess_dataset(dataset=dataset)
    hp = get_dataset_dict(hp_dict, dataset.name)
    if hp is None:
        print("No hyperparameters found for dataset: {0}".format(dataset.name))
        continue

    snn_model = snn.SNN(epochs=100, hp=hp)
    result, history = snn_model.analyze(dataset, data, hp)

    columns = list(
        ["dataset", "rows", "features", "task", "model_name", "test_score", "test_accuracy", "hyperparameters"])
    result = pd.DataFrame(data=[result],  # values
                          columns=columns)

    save_dataset_results(dataset, result)
    save_time_measure(dataset.name, start_time, time.time())
    create_new_tf_session()
