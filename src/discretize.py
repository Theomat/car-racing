import numpy as np


MAX_ACTION: int = 5

TURN_LEFT = 0
TURN_RIGHT = 1
BRAKE = 2
ACCELERATE = 3
DO_NOTHING = 4

def action_discrete2continous(action: int) -> np.ndarray:
    """
    Transform a discrete action to a triple (s, t, b).
    s in [-1;1] is the steering angle
    t in [0;1] is the throttle
    b in [0;1] is the brake
    """
    if action == TURN_LEFT:
        return np.array([-1.0, 0.0, 0.0])
    elif action == TURN_RIGHT:
        return np.array([+1.0, 0.0, 0.0])
    elif action == BRAKE:
        return np.array([0.0, 0.0, +0.8])
    elif action == ACCELERATE:
        return np.array([0.0, +1.0, +0.8])
    elif action == DO_NOTHING:
        return np.array([0.0, 0.0, 0.0])
    else:
        print("Action outside of action space !")



def action_random() -> int:
    return np.random.randint(0, 5)
