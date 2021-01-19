from typing import Callable

import numpy as np

Annealing = Callable[[int], float]


def constant(constant: float) -> Annealing:
    return lambda i: constant


def linear(start: float, end: float, duration: int) -> Annealing:
    return lambda i: start + (end - start) * i / duration


def exponential(start: float, end: float, decay: float) -> Annealing:
    return lambda i: end + (start - end) * np.exp(- i / decay)


def translated(translation: int, annealing: Annealing) -> Annealing:
    return lambda i: annealing(i + translation)
