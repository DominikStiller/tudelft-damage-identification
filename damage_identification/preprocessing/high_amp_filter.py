import matplotlib.pyplot as plt
import numpy as np

from damage_identification.io import load_compressed_data

class HighAmpFiltering():

    def __init__(self):
        return


    def filter_signal(self, data):
        signal = data
        idx = []
        for i in range(0, len(signal)-1):
            x = max(signal[i, :])
            if x > 0.0995:
                idx.append(i)
        for j in idx:
            data = np.delete(data, j, axis=0)
        return data


    def plot(self, data, n):
        plt.figure(figsize=(10, 6))
        plt.plot(data[n, :], label="Raw", color="b")
        plt.show()


if __name__ == "__main__":
    filter = HighAmpFiltering()
    signal = load_compressed_data("data/comp0.tradb")
    new_signal = filter.filter_signal(signal)
    filter.plot(new_signal,555)
