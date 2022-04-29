import pywt
from damage_identification import io


class MultiResolutionAnalysis:
    """
    This class extracts the signal data to decompose into energy levels at specified time bands and carrier frequencies.

    """

    # def __init__(self, params: Dict[str: Any]):
    def __init__(self, wavelet, mode, time_bands, dec_level):
        """
        Initializes object with specifications for decomposing signal.

        Parameters:
            - wavelet: the name and magnitude of the wavelet family (e.g.: 'db38' or 'coif17')
            - mode: the decomposition mode used (e.g.: 'symmetric' or )
            - time_bands: the amount of time bands to split the energy information (i.e. with time_bands = 4, the energy information describes the 4 quarters of the signal)
            - dec_level: the decomposition level of the signal (i.e. the amount of carrier frequencies the signal will be split up into. e.g.: decomposition level of 4 gives 2^4 = 16 frequencies)
        """
        self.signal_data = []
        self.wavelet = wavelet
        self.mode = mode
        self.time_bands = time_bands
        self.dec_level = dec_level
        self.wave_coeffs = []

    def load(self, directory, n):
        """
        Loads compressed data from external file.

        Parameters:
            - directory: directory of data file
            - n: the waveform that is to be decomposed
        """
        x = io.load_compressed_data(directory)
        self.signal_data = x[n, :]
        return self.signal_data

    def load_manual(self, data):
        """
        Loads manually input data array into signal data
        (for testing purposes)

        Parameters:
            - data: custom signal data in array
        """
        self.signal_data = data
        return self.signal_data

    def data_handler(self):
        """
        Decomposes data in object and flattens binary tree into three-dimensional array.

        Parameters:
            - None
        """
        wp = pywt.WaveletPacket(data=self.signal_data, wavelet=self.wavelet, mode=self.mode)
        for level in range(1, wp.maxlevel+1):
            self.wave_coeffs.append([wp[node.path].data for node in wp.get_level(level, 'natural')])

        return self.wave_coeffs

    def decomposer(self):
        """
        Decomposes frequency band energy data into frequency and time band data.

        Parameters:
            - None
        """
        energy = []
        time_energy = []
        c = 0
        tot_energy = sum(self.wave_coeffs[0][0][:]**2)+sum(self.wave_coeffs[0][1][:]**2)
        for i in range(0, 2**self.dec_level):
            # print(len(coeffs[self.dec_level-1][i]))
            # print(coeffs[self.dec_level-1][i][:])
            for t in range(0, self.time_bands):
                left_t_bound = int((t/self.time_bands)*len(self.wave_coeffs[self.dec_level-1][i][:]))
                # print(i,t)
                # print(coeffs[self.dec_level-1][i][:])
                right_t_bound = int((((t+1)/self.time_bands)*len(self.wave_coeffs[self.dec_level-1][i][:])))
                time_energy.append(sum(self.wave_coeffs[(self.dec_level-1)][i][left_t_bound:right_t_bound]**2)/tot_energy)

            energy.append(time_energy)
            time_energy = []

        for j in range(0, 2**self.dec_level):
            c = c + sum(energy[j][:])

        return energy, c
