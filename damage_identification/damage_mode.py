from enum import Enum, auto


class DamageMode(Enum):
    UNKNOWN = auto()  # Damage mode of cluster could not be identified
    INVALID = auto()  # Example is invalid (e.g. all data removed through filtering)
    DELAMINATION = auto()
    MATRIX_CRACKING = auto()
    FIBER_MATRIX_DEBONDING = auto()
    FIBER_FRACTURE = auto()
    FIBER_PULLOUT = auto()
