from typing import Callable

import numpy as np

Annealing = Callable[[int], float]


def constant(constant: float) -> Annealing:
    return lambda i: constant


def linear(start: float, end: float, duration: int) -> Annealing:
    return lambda i: start + (end - start) * i / duration


def exponential(start: float, end: float, decay: float) -> Annealing:
    return lambda i: start + (end - start) * np.exp(- i / decay)
