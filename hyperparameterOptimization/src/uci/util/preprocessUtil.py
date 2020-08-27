from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src.uci.entity.datasetConfig import DatasetConfig
from src.uci.util.datasetUtil import load_dataset
from src.uci.util.task_enum import TaskType

text_dict = dict()
test_size = 0.15


def preprocess_dataset(dataset: DatasetConfig):
    '''

    :param dataset:
    :return:  x_train, x_test, y_train, y_test
    '''
    x, y = load_dataset(dataset)
    dataset.features = x.shape[1]
    dataset.rows = x.shape[0]

    # set amount of in_params and out_params
    dataset.n_in = x.shape[1]

    if dataset.task == TaskType.CLASSIFICATION:
        y = to_categorical(y)
        dataset.n_out = y.shape[1]
        print('TASK CLASSIFICATION output layer size {0}'.format(dataset.n_out))
    elif dataset.task == TaskType.REGRESSION:
        dataset.n_out = 1
        print('TASK REGRESSION output layer size {0}'.format(dataset.n_out))
    elif dataset.task == TaskType.LOGISTIC_REGRESSION:
        dataset.n_out = y.shape[1]
        print('TASK LOGISTIC REGRESSION output layer size {0}'.format(dataset.n_out))
    else:
        raise Exception("Task not known{0}".format(dataset.task))

    return train_test_split(x, y, test_size=test_size, shuffle=True)
