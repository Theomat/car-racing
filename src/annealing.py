from typing import Callable

import numpy as np

Annealing = Callable[[int], float]


def constant(constant: float) -> Annealing:
    return lambda i: constant


def linear(start: float, end: float, duration: int) -> Annealing:
    return lambda i: max(start + (end - start) * i / duration, end)


def exponential(start: float, end: float, decay: float) -> Annealing:
    return lambda i: max(end + (start - end) * np.exp(- i / decay), end)


def translated(translation: int, annealing: Annealing) -> Annealing:
    return lambda i: annealing(i + translation)
