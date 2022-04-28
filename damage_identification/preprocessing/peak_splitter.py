import numpy as np


class PeakSplitter:
    """
    Splits a waveform if it contains two signals. Multiple signals are detected based on peaks.
    """

    def __init__(self, waveform, lag=160, threshold=4, influence=1, threshold_counter=5):
        self.y = list(np.zeros(lag + 1))
        self.waveform = waveform
        self.threshold_counter = threshold_counter
        self.length = len(self.y)
        self.counter = np.zeros(self.length + len(self.waveform))
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0 : self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0 : self.lag]).tolist()

    def split(self):
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
                return 0
            elif i == self.lag:
                self.signals = [0] * len(self.y)
                self.filteredY = np.array(self.y).tolist()
                self.avgFilter = [0] * len(self.y)
                self.stdFilter = [0] * len(self.y)
                self.avgFilter[self.lag] = np.mean(self.y[0 : self.lag]).tolist()
                self.stdFilter[self.lag] = np.std(self.y[0 : self.lag]).tolist()
                return 0

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
