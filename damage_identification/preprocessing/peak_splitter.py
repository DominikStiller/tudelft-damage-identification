import numpy as np
from tqdm import tqdm


class PeakSplitter:
    """
    Splits a waveform if it contains two signals. Multiple signals are detected based on peaks.
    """

    def __init__(
        self,
        waveform: np.ndarray,
        lag=160,
        threshold=4,
        influence=1,
        threshold_counter=5,
        window=50,
    ):
        self.waveform = waveform
        self.lag = lag
        self.window = window
        self.threshold = threshold
        self.influence = influence
        self.threshold_counter = threshold_counter

        self.y = list(np.zeros(self.lag + 1))
        self.length = len(self.y)
        self.counter = np.zeros(self.length + len(self.waveform))
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0 : self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0 : self.lag]).tolist()

    def split_single(self) -> list[np.ndarray]:
        """
        Detects peaks and splits waveform if there are two peaks.

        Based on https://stackoverflow.com/a/56451135

        This method is slow (about 60 ms per example with default settings). Therefore, it is
        infeasible to actually use this in practice. Profiling shows that std and sum applied
        to all sliding window views is the main bottleneck.

        Returns:
            A tuple of two signals, or the original signal and None
        """
        padded_waveform = np.pad(
            abs(self.waveform), (self.lag - 1, 0), "constant", constant_values=(0, 0)
        )
        windows_waveform = np.lib.stride_tricks.sliding_window_view(padded_waveform, self.lag)
        threshold_stds = np.apply_along_axis(np.std, 1, windows_waveform) * self.threshold
        self.signal = np.less(threshold_stds, abs(self.waveform)).astype(int)
        windows_signal = np.lib.stride_tricks.sliding_window_view(self.signal, self.window)
        signal = np.apply_along_axis(np.sum, 1, windows_signal)
        indexes = np.where(signal > self.threshold_counter)[0]
        pad_left = np.pad(indexes, (1, 0), "constant", constant_values=(0, 0))
        pad_right = np.pad(indexes, (0, 1), "constant", constant_values=(0, 0))
        consecutives = np.where(pad_right == pad_left + 1)[0]
        indexes = np.delete(indexes, consecutives)
        indexes = np.delete(
            indexes, np.where(indexes < len(self.waveform) * 0.2)
        )  # Select first 20% of waveform as single hit
        indexes = np.concatenate((np.array([0]), indexes, np.array([len(self.waveform)])))
        slices = []
        for i in range(len(indexes) - 1):
            slice = self.waveform[indexes[i] : indexes[i + 1]]
            slice = np.pad(
                slice, [0, len(self.waveform) - len(slice)], mode="constant", constant_values=0
            )
            slices.append(slice)
        return slices

    @staticmethod
    def split_all(data: np.ndarray) -> tuple[np.ndarray, int, int, int]:
        """
        Split all examples in a dataset if the example contains two peaks

        Args:
            data: the dataset (shape n_examples x n_samples)

        Returns:
            A tuple of the processed dataset, and the numbers of examples with no peaks, one peak and two peaks
        """
        examples = []

        n_no_peaks = 0
        n_one_peak = 0
        n_over_two_peaks = 0

        n_examples = data.shape[0]

        with tqdm(total=n_examples, file=sys.stdout) as pbar:
            for i in range(n_examples):
                slices = PeakSplitter(data[i]).split_single()
                examples.extend(slices)

                n_slices = len(slices)

                if n_slices == 0:
                    n_no_peaks += 1
                elif n_slices == 1:
                    n_one_peak += 1
                else:
                    n_over_two_peaks += 1

                pbar.update()

        return np.vstack(examples), n_no_peaks, n_one_peak, n_over_two_peaks


if __name__ == "__main__":
    import os
    import sys

    from damage_identification.io import load_data_from_csv, load_data_from_tradb

    filename = sys.argv[1]
    filename_split = os.path.splitext(filename)[0] + "_split.npy"

    print("Loading data...")
    if filename.endswith(".csv"):
        data = load_data_from_csv(filename)
    elif filename.endswith(".tradb"):
        data = load_data_from_tradb(filename)
    else:
        raise Exception("Unsupported data file type")

    print("Splitting data...")
    data_split, n_no_peaks, n_one_peak, n_over_two_peaks = PeakSplitter.split_all(data)

    print(f"Dataset with originally {data.shape[0]} contains")
    print(f" - {n_no_peaks} examples without peaks")
    print(f" - {n_one_peak} examples with one peak peaks")
    print(f" - {n_over_two_peaks} examples with two peaks or more")

    np.save(filename_split, data_split)
    print(f"Saved split data of {data_split.shape[0]} examples to {filename_split}")
