import numpy as np

from typing import Tuple


ACTION_SPACE: Tuple[np.ndarray] = (
    np.array([-1, 1, 0.]),
    np.array([-1, 1, .2]),
    np.array([-1, 0, 0.]),
    np.array([-1, 0, .2]),
    np.array([0, 1, 0.]),
    np.array([0, 1, .2]),
    np.array([0, 0, 0.]),
    np.array([0, 0, .2]),
    np.array([1, 1, 0.]),
    np.array([1, 1, .2]),
    np.array([1, 0, 0.]),
    np.array([1, 0, .2]),
)
MAX_ACTION: int = len(ACTION_SPACE)


def action_discrete2continous(action: int) -> np.ndarray:
    """
    Transform a discrete action to a triple (s, t, b).
    s in [-1;1] is the steering angle
    t in [0;1] is the throttle
    b in [0;1] is the brake
    """
    if action < MAX_ACTION:
        return ACTION_SPACE[action]
    else:
        print("Action outside of action space !")


def action_random() -> int:
    return np.random.randint(0, MAX_ACTION)
