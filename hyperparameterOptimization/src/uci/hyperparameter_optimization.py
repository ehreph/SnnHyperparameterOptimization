import keras as K
import tensorflow as tf
from itertools import product

from numpy.ma import array

from src.uci.network import snn
from src.uci.util.datasetUtil import load_datasets_config
from src.uci.util.preprocessUtil import preprocess_dataset
from src.uci.util.util import create_results_table, save_dataset_results
from src.uci.util.hyperparameterUtil import load_hyperparameter_configuration
# Configure Tensorflow and Keras to run with GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 16})
config.gpu_options.allow_growth = True

epochs = 100
data_config = load_datasets_config('datasets_config_complete')
hp = load_hyperparameter_configuration('snn')
load_models = True
save_models = True

# generate all permutations of the given hyperparameters
permutations = [dict(zip(hp, v)) for v in product(*hp.values())]
assert len(permutations) > 0  # at least 1 hyperparameter configuration is needed


def create_new_TF_session():
    K.backend.clear_session()
    s = tf.Session(config=config)
    K.backend.set_session(s)


# create initial KERAS session
create_new_TF_session()

for dataset in data_config:
    data = preprocess_dataset(dataset=dataset)
    snn_model = snn.SNN(epochs=epochs, hp=hp)
    print("start hyperparameter optimization for dataset {0} ".format(dataset.name))

    result, history = snn_model.analyze_set(dataset, *data, permutations, save=save_models, load=load_models)

    result = create_results_table(array(result))
    save_dataset_results(dataset, result)
    print("finish hyperparameter optimization for dataset {0} ".format(dataset.name))
    create_new_TF_session()
