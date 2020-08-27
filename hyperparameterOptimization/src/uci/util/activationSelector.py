from enum import Enum


class ActivationType(Enum):
    selu = 1


def get_activation_type(activation):
    if activation == ActivationType.selu.value or activation == ActivationType.selu.name:
        return 'selu'
    else:
        return "relu"
