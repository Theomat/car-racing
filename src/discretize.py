from enum import IntFlag


import numpy as np


class DiscreteAction(IntFlag):
    NOTHING = 0
    ACCELERATE_SLOW = 1
    ACCELERATE_MID = ACCELERATE_SLOW << 1
    ACCELERATE_FAST = ACCELERATE_SLOW + ACCELERATE_MID
    BRAKE_SLOW = ACCELERATE_MID << 1
    BRAKE_MID = BRAKE_SLOW << 1
    BRAKE_FAST = BRAKE_SLOW + BRAKE_MID
    TURN_LEFT_SLOW = BRAKE_MID << 1
    TURN_LEFT_MID = TURN_LEFT_SLOW << 1
    TURN_LEFT_FAST = TURN_LEFT_SLOW + TURN_LEFT_MID
    TURN_RIGHT_SLOW = TURN_LEFT_MID << 1
    TURN_RIGHT_MID = TURN_RIGHT_SLOW << 1
    TURN_RIGHT_FAST = TURN_RIGHT_SLOW + TURN_RIGHT_MID

    def has_flag(self, flag: int) -> bool:
        return self & flag == flag


MAX_ACTION: int = DiscreteAction.TURN_RIGHT_FAST + DiscreteAction.BRAKE_FAST


def action_discrete2continous(action: int) -> np.ndarray:
    """
    Transform a discrete action to a triple (s, t, b).
    s in [-1;1] is the steering angle
    t in [0;1] is the throttle
    b in [0;1] is the brake
    """
    s, t, b = 0, 0, 0
    action: DiscreteAction = DiscreteAction(action)
    if action.has_flag(DiscreteAction.ACCELERATE_SLOW):
        t += 0.33
    if action.has_flag(DiscreteAction.ACCELERATE_MID):
        t += 0.66
    if action.has_flag(DiscreteAction.BRAKE_SLOW):
        b += 0.33
    if action.has_flag(DiscreteAction.BRAKE_MID):
        b += 0.66
    if action.has_flag(DiscreteAction.TURN_LEFT_SLOW):
        s -= 0.33
    if action.has_flag(DiscreteAction.TURN_LEFT_MID):
        s -= 0.66
    if action.has_flag(DiscreteAction.TURN_RIGHT_SLOW):
        s += 0.33
    if action.has_flag(DiscreteAction.TURN_RIGHT_MID):
        s += 0.66
    return np.array([s, t, b])
