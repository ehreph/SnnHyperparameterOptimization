from enum import Enum

from keras.layers import AlphaDropout, Dropout


class DropoutType(Enum):
    ALPHA_DROPOUT = "AlphaDropout",
    DROPOUT = "Dropout",
    DROP_TO_VALUE = "dropToValue"


def get_dropout_class(dropout):
    if dropout == DropoutType.ALPHA_DROPOUT.value or dropout == DropoutType.DROP_TO_VALUE.value:
        return AlphaDropout
    else:
        return Dropout
