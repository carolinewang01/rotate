from enum import Enum

class Action(Enum):
    NONE = 0  # None
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5

NUM_ACTIONS = len(Action)
print(NUM_ACTIONS)  # 6