import numpy as np
import pandas as pd



class Normalization():

    def __init__(self, train_data):
        self.train_data = train_data
        self.bounds = None


    def save(self, directory):
        """
        Saves

        Args:
            directory: the directory to save the state to
        """

        self.bounds.tocsv

        pass


    def load(self, directory):
        """
        Loads

        Args:
            directory: the directory to load the state from
        """
        pass


    def train(self):
        """
        Train

        Args:
            examples: the training set with all training examples (shape n_examples x length_example)
        """

        columns = pd.DataFrame({
            "first_n_samples": [],
            "peak_amplitude": [],
            "counts": [],
            "duration": [],
            "rise_time": [],
            "energy": []
            })
        self.bounds = pd.DataFrame(columns,
                              index = pd.Index(["min", "max"]))
        for column in self.train_data.columns:
            self.bounds.loc["max", column] = self.train_data[column].max()
            self.bounds.loc["min", column] = self.train_data[column].min()

        Normalization.save()

    def transform(self, data):
         """
         Extracts features from a single waveform.

         Args:
             example: a single example (shape 1 x length_example)

         Returns:
             A dictionary containing items with each feature name value for the input example.
             Example: {"duration": 30.4, "average_amplitude": 3.7}
             An example can be marked as invalid by setting at least one of the features to None.
         """




testdata = pd.DataFrame({
            "first_n_samples": np.linspace(0,1,10),
            "peak_amplitude": np.linspace(0,10,10),
            "counts": np.linspace(-50,100, 10),
            "duration": np.linspace(100,1000, 10),
            "rise_time": np.linspace(1000,10000, 10),
            "energy": np.linspace(-1000,1000, 10)
            })
nrml = Normalization(testdata)
nrml.train()
print(nrml.bounds)
