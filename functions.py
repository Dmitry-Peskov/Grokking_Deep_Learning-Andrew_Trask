import numpy as np
from typing import Sequence


def weighted_amount(
        features: Sequence[int | float],
        weights: Sequence[int | float]
) -> int | float:
    """
    Получить взвешанную сумму.

    :param features: последовательность свойств / параметров
    :param weights: последовательность весов
    :return: взвешанная сумма входов
    """
    assert (len(features) == len(weights)), "Количество элементов в features и weights должно быть одинаковым"
    return sum([features[idx] * weights[idx] for idx in range(len(features))])


def element_multiplication(
        features: int | float,
        weights: Sequence[int | float]
) -> np.ndarray:
    """
    Произвести поэлементное умножение некоторого свойства, на каждый вес из массива weights

    :param features: некоторое свойство
    :param weights: массив содержащий веса
    :return: массив произведений свойства на вес
    """
    output = np.array([0 for _ in range(len(weights))])
    for idx in range(len(weights)):
        output[idx] = features * weights[idx]
    return output
