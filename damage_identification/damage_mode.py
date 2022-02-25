from enum import Enum, auto


class DamageMode(Enum):
    UNKNOWN = auto()
    DELAMINATION = auto()
    MATRIX_CRACKING = auto()
    FIBER_MATRIX_DEBONDING = auto()
    FIBER_FRACTURE = auto()
    FIBER_PULLOUT = auto()
