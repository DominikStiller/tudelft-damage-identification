import numpy as np
import pandas as pd
import os

class Normalization():
    """
    This class normalizes all the features from the direct feature extraction.

    The train function is used with a training dataset to 

    """
    def __init__(self):
        self.bounds = None


    def save(self, directory):
        """
        Saves

        Args:
            directory: the directory to save the state to
        """
        self.bounds.to_csv(directory)

        pass


    def load(self, directory):
        """
        Loads

        Args:
            directory: the directory to load the state from
        """
        self.bounds = pd.read_csv(directory)

        pass


    def train(self, train_data):
        """
        Creates bounds based on some training data

        Args:
            train_data: pandas dataframe of the training data

        Returns:
            bounds: pandas dataframe containing the min and max of each column of train_data
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
        for column in train_data.columns:
            self.bounds.loc["max", column] = train_data[column].max()
            self.bounds.loc["min", column] = train_data[column].min()

        #Normalization.save()

    def transform(self, data):
        """
        Transforms data based on the bounds from training data

        Args:
            data: data to be transformed

        Returns:
            normalize_data: pandas dataframe of the input data normalize between -1 and 1
        """

        normalize_data = 2*(data - self.bounds.min())/(self.bounds.max() - self.bounds.min()) - 1
        return normalize_data



testdata = pd.DataFrame({
            "first_n_samples": np.linspace(0,1,10),
            "peak_amplitude": np.linspace(0,10,10),
            "counts": np.linspace(-50,100, 10),
            "duration": np.linspace(100,1000, 10),
            "rise_time": np.linspace(1000,10000, 10),
            "energy": np.linspace(-1000,1000, 10)
            })

print(s(np.random.rand(1, 10)))

randomset = pd.DataFrame({
            "first_n_samples": np.random.rand(1, 10),
            "peak_amplitude": np.random.rand(1, 10)*10,
            "counts": np.random.rand(1, 10)*150 - 50,
            "duration": np.random.rand(1, 10)* 1000 + 100,
            "rise_time": np.random.rand(1, 10)*90000 + 1000,
            "energy": np.random.rand(1, 10)*2000 -1000
            })
print(randomset['duration'])


nrml = Normalization()
nrml.train(testdata)


print(nrml.transform(randomset))
