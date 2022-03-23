from enum import Enum, auto


class DamageMode(Enum):
    UNKNOWN = auto()  # Clustering could not assign example to a cluster
    INVALID = auto()  # Example is invalid (e.g. two damages in example)
    DELAMINATION = auto()
    MATRIX_CRACKING = auto()
    FIBER_MATRIX_DEBONDING = auto()
    FIBER_FRACTURE = auto()
    FIBER_PULLOUT = auto()
