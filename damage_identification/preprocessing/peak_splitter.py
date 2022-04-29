import sys
import time
from typing import Optional

import numpy as np
from tqdm import tqdm

from damage_identification.io import load_uncompressed_data


class PeakSplitter:
    """
    Splits a waveform if it contains two signals. Multiple signals are detected based on peaks.
    """

    def __init__(
        self, waveform: np.ndarray, lag=160, threshold=4, influence=1, threshold_counter=5
    ):
        self.waveform = waveform
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.threshold_counter = threshold_counter
        self._reset()

    def _reset(self):
        self.y = list(np.zeros(self.lag + 1))
        self.length = len(self.y)
        self.counter = np.zeros(self.length + len(self.waveform))
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0 : self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0 : self.lag]).tolist()

    def split_single(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detects peaks and splits waveform if there are two peaks.

        Based on https://stackoverflow.com/a/56451135

        Returns:
            A tuple of two signals, or the original signal and None
        """
        for item in abs(self.waveform):
            self.y.append(item)
            i = len(self.y) - 1
            if i < self.lag:
                return None, None
            elif i == self.lag:
                self.signals = [0] * len(self.y)
                self.filteredY = np.array(self.y).tolist()
                self.avgFilter = [0] * len(self.y)
                self.stdFilter = [0] * len(self.y)
                self.avgFilter[self.lag] = np.mean(self.y[0 : self.lag]).tolist()
                self.stdFilter[self.lag] = np.std(self.y[0 : self.lag]).tolist()
                return None, None

            self.signals += [0]
            self.filteredY += [0]
            self.avgFilter += [0]
            self.stdFilter += [0]

            if abs(self.y[i] - self.avgFilter[i - 1]) > self.threshold * self.stdFilter[i - 1]:
                if self.y[i] > self.avgFilter[i - 1]:
                    self.signals[i] = 1
                else:
                    self.signals[i] = -1

                self.filteredY[i] = (
                    self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]
                )
                self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag) : i])
                self.stdFilter[i] = np.std(self.filteredY[(i - self.lag) : i])
            else:
                self.signals[i] = 0
                self.filteredY[i] = self.y[i]
                self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag) : i])
                self.stdFilter[i] = np.std(self.filteredY[(i - self.lag) : i])

            if self.signals[i] == 1 and i != 0:
                self.counter[i] = self.counter[i - 1] + 1
            if (
                i > self.lag + 0.2 * len(self.waveform)
                and self.counter[i] >= 1
                and sum(self.counter[i - 50 : i]) > self.threshold_counter
            ):
                indexslice = int(0.95 * (i - self.lag))
                firstslice = self.waveform[:indexslice]
                firstslice = np.pad(
                    firstslice,
                    (0, len(self.waveform) - indexslice),
                    mode="constant",
                    constant_values=0,
                )
                secondslice = self.waveform[indexslice:]
                secondslice = np.pad(
                    secondslice, (0, indexslice), mode="constant", constant_values=0
                )
                return firstslice, secondslice
            i = i + 1
        return self.waveform, None

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
        n_two_peaks = 0

        n_examples = data.shape[0]

        with tqdm(total=n_examples, file=sys.stdout) as pbar:
            for i in range(n_examples):
                first, second = PeakSplitter(data[i]).split_single()

                if first is not None:
                    examples.append(first)
                if second is not None:
                    examples.append(second)

                if (first is not None) and (second is not None):
                    n_two_peaks += 1
                elif (first is None) and (second is None):
                    n_no_peaks += 1
                else:
                    n_one_peak += 1

                pbar.update()

        return np.vstack(examples), n_no_peaks, n_one_peak, n_two_peaks


if __name__ == '__main__':
    data = load_uncompressed_data("data/Waveforms.csv")[0]
    data_big = np.vstack([data] * 100)

    start = time.perf_counter()
    PeakSplitter.split_all(data_big)
    print(f"split_all: {time.perf_counter() - start:.3f} s")

    start = time.perf_counter()
    for _ in range(100):
        PeakSplitter(data).split_single()
    print(f"split_single: {time.perf_counter() - start:.3f} s")

    start = time.perf_counter()
    splt = PeakSplitter(data)
    for _ in range(100):
        splt._reset()
        splt.split_single()
    print(f"split_single reuse: {time.perf_counter() - start:.3f} s")
