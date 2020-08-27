from src.uci.entity.datasetConfig import DatasetConfig
from src.uci.util.activationSelector import get_activation_type
from src.uci.util.dropoutSelector import get_dropout_class
from src.uci.util.task_enum import TaskType
from src.uci.util.util import save_network, load_network_set, get_next_layer_size

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.noise import AlphaDropout
from keras.optimizers import SGD
from keras import losses


class SNN:

    def __init__(self, batch_size=32, epochs=100, test_size=0.15, name='snn', hp={}):
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_size = test_size
        self.name = name
        self.hyper_parameters = hp

    def set_hyper_parameters(self, hp):
        self.hyper_parameters = hp

    def create_network(self,
                       dataset: DatasetConfig,
                       nstart=16,
                       nlayers=6,
                       activationFn='selu',
                       dropoutFn=AlphaDropout,
                       dR=0.1,
                       learningrate=0.01,
                       layerForm='rect'):
        """Generic function to create a fully-connected neural network.

        # Arguments
            dataset: DatasetConfig contains information of dataset
            nstart: int > 0. Number of dense units per layer.
            nlayers: int > 0. Number of dense layers.
            dropoutFn: str or enum of type DropoutType.
            dR: 0 <= float <= 1. The rate of dropout.
            learningrate: 0 <= float <= 1. The rate of learning.
            layerForm: layer form of model
        # Returns
            A Keras model instance (compiled).
        """

        optimizer = 'sgd'
        kernel_initializer = 'lecun_normal'
        dropout_instance = get_dropout_class(dropoutFn)
        activation_type = get_activation_type(activationFn)

        model = Sequential()
        model.add(Dense(units=nstart, input_shape=(dataset.n_in,),
                        kernel_initializer=kernel_initializer, activation=activation_type))
        model.add(AlphaDropout(rate=dR))

        if layerForm == 'rect':
            for i in range(1, nlayers):
                model.add(Dense(nstart, kernel_initializer=kernel_initializer, activation=activation_type))
                model.add(dropout_instance(rate=dR))
        elif layerForm == 'cone':
            units = nstart
            for i in range(1, nlayers):
                #  calculate number of neurons for the next layer
                units = get_next_layer_size(dataset.n_out, dataset.n_in, nlayers - i, units)
                model.add(Dense(units, kernel_initializer=kernel_initializer, activation=activation_type))
                model.add(dropout_instance(rate=dR))

        opt = None
        if optimizer == 'sgd':
            opt = SGD(lr=learningrate)

        if dataset.task == TaskType.CLASSIFICATION:
            print("TASK: CLASSIFICATION")
            if dataset.n_out == 2:
                model.add(Dense(units=dataset.n_out, activation="sigmoid"))
                model.compile(loss=losses.binary_crossentropy, optimizer=opt,
                              metrics=['accuracy'])
            else:
                model.add(Dense(units=dataset.n_out, activation="sigmoid"))
                model.compile(loss=losses.categorical_crossentropy, optimizer=opt,
                              metrics=['accuracy'])

        elif dataset.task == TaskType.REGRESSION:
            print("TASK: REGRESSION")
            assert dataset.n_out == 1
            model.add(Dense(units=dataset.n_out, activation="linear"))
            model.compile(loss=losses.mean_squared_error, optimizer=opt, metrics=['mae'])

        elif dataset.task == TaskType.LOGISTIC_REGRESSION:
            print("TASK: LOGISTIC REGRESSION")
            model.add(Dense(units=dataset.n_out, activation="sigmoid"))
            model.compile(loss='categorical_crossentropy', optimizer=opt)

        return model

    def compile_network_set(self, dataset, permutations, start_index=0):
        for hp in permutations[start_index:]:
            yield self.create_network(dataset, **hp)

    def analyze_set(self, dataset: DatasetConfig, x_train, x_test, y_train, y_test, permutations, load,
                    save=True):

        histories: list = list()
        result: list = list()

        # load existing models
        models: [] = []
        if load:
            models.extend(load_network_set(dataset, self.name, len(permutations)))

        # compile rest of permutation set
        if len(models) == 0 or len(models) < len(permutations):
            models.extend(self.compile_network_set(dataset, permutations, start_index=len(models)))

        # train models
        histories.extend(
            self.train_network_set(x_train, y_train, models, dataset, save))

        # evaluate trained models
        result.extend(
            self.evaluate_network_set(x_test, y_test, models, permutations, dataset=dataset))
        return result, histories

    def train_network_set(self, x_train, y_train, models,
                          current_dataset, save):
        for index, model in enumerate(models, start=0):
            history = model.fit(x=x_train,
                                y=y_train,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=1,
                                validation_split=self.test_size)

            if save:
                save_network(model, current_dataset, self.name + str(index))
            yield history
        print("finished Training models")

    def analyze(self, dataset: DatasetConfig, data, hp):
        x_train, x_test, y_train, y_test = data
        # compile model wit hyperparemeters
        model = self.create_network(dataset, **hp)

        # train model
        history = model.fit(x=x_train,
                            y=y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=2,
                            validation_split=self.test_size)

        # evaluate trained model
        score = model.evaluate(x=x_test,
                               y=y_test,
                               batch_size=self.batch_size,
                               verbose=1)

        model_name = self.name + '_' + str(dataset.name)
        print('Netork {0} results: '.format(model_name))
        if dataset.task == TaskType.CLASSIFICATION:
            print('Test score:', score[0])
            print('Test accuracy:', score[1])
            result = [dataset.name, dataset.rows, dataset.features, dataset.task.name, model_name, score[0], score[1],
                      hp]

        return result, history

    def evaluate_network_set(self, x_test, y_test, models, permutations, dataset):
        '''
        evaluates all models


        :param x_test:
        :param y_test:
        :param batch_size:
        :param models:
        :param permutations:
        :param base_model_name:
        :param dataset:
        :return:  returns pandas evaluation table
        '''

        assert len(models) == len(permutations)

        for index, trained_model in enumerate(models, start=0):
            model_name = self.name + str(index)
            perm = permutations[index]
            score = trained_model.evaluate(x_test,
                                           y_test,
                                           batch_size=self.batch_size,
                                           verbose=1)

            print('Netork {0} results: '.format(model_name))
            if dataset.task == TaskType.CLASSIFICATION:
                print('Test score:', score[0])
                print('Test accuracy:', score[1])
                yield [dataset.name, dataset.rows, dataset.features, dataset.task.name, model_name, score[0], score[1],
                       perm]
            elif dataset.task == TaskType.REGRESSION:
                print('score {0}'.format(score))
                yield [dataset.name, dataset.rows, dataset.features, dataset.task.name, model_name, "", "", perm]

            elif dataset.task == TaskType.LOGISTIC_REGRESSION:
                print('score {0}'.format(score))
                yield [dataset.name, dataset.rows, dataset.features, dataset.task.name, model_name, 0, 0, perm]
        return
