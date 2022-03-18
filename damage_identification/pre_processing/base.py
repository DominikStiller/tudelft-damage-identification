from abc import ABC, abstractmethod
from typing import Dict, Any


class PreProcessing(ABC):
    """
    A base class for all pre-processing methods.
    """

    @abstractmethod
    def __init__(self, name: str, params: Dict[str, Any]):
        """
        Initialize the pre-processing method.

        Args:
            name: name of the pre-processing method
            params: parameters for the pre-processing method
        """
        self.name = name
        self.params = params
