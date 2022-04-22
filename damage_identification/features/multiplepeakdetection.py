import numpy as np


class real_time_peak_detection:
    def __init__(self, array, lag=80, threshold=4, influence=1):
        self.y = list(array)
        self.length = len(self.y)
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0 : self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0 : self.lag]).tolist()

    def thresholding_algo(self, new_value):
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)
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

        return self.signals[i], self.stdFilter[i], self.avgFilter[i]

    def test_peak(self, lag=80, threshold=4, influence=1) -> dict[str, np.ndarray]:
        test = abs(np.array(self.y))
        real_time_test = real_time_peak_detection(
            test, lag, threshold, influence
        )  # possibly increase lag
        threshold_counter = 5
        i = 0
        signal = [0] * len(test)
        std_filter = [0] * len(test)
        avg_filter = [0] * len(test)
        counter = np.zeros(len(test))
        for element in test:
            next_signal, next_std, next_avg = real_time_test.thresholding_algo(element)
            signal[i] = next_signal
            std_filter[i] = next_std
            avg_filter[i] = next_avg
            if signal[i - 1] == 1 and i != 0:
                counter[i] = counter[i - 1] + 1

            if (
                i > 0.2 * self.length
                and counter[i] >= 1
                and sum(counter[i - int(0.024 * self.length) : i]) > threshold_counter
            ):  # possibly decrease range
                indexslice = int(0.95 * i)
                firstslice = test[:indexslice]
                firstslice = np.pad(
                    firstslice, [0, self.length - indexslice], mode="constant", constant_values=0
                )
                secondslice = test[indexslice:]
                secondslice = np.pad(
                    secondslice, [0, indexslice], mode="constant", constant_values=0
                )
                print(np.shape(secondslice))
                return {"firstslice": firstslice, "secondslice": secondslice}
            i = i + 1
        return {"firstslice": test, "secondslice": None}
