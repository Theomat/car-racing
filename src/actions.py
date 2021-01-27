LEFT = [-1.0, 0.0, 0.0]
RIGHT = [+1.0, 0.0, 0.0]
ACCELERATE = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]
NOTHING = [0.0, 0.0, 0.0]


ACTIONS = [LEFT, RIGHT, ACCELERATE, BRAKE, NOTHING]
ACTION_SPACE = len(ACTIONS)

def discrete2cont(action):
    return ACTIONS[action]
