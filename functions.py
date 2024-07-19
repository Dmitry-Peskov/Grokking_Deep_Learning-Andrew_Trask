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
