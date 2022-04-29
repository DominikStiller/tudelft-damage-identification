from typing import Dict, Tuple
import pywt
from damage_identification import io
import warnings


class MultiResolutionAnalysis:
    """
    This class extracts the signal data to decompose into energy levels at specified time bands and carrier frequencies.

    """

    # def __init__(self, params: Dict[str: Any]):
    def __init__(self, wave_fam: str, wave_mag: int, time_bands: int, dec_level: int):
        """
        Initializes object with specifications for decomposing signal.

        Parameters:
            - wavelet: the name and magnitude of the wavelet family (e.g.: 'db38' or 'coif17')
            - time_bands: the amount of time bands to split the energy information (i.e. with time_bands = 4, the energy information describes the 4 quarters of the signal)
            - dec_level: the decomposition level of the signal (i.e. the amount of carrier frequencies the signal will be split up into. e.g.: decomposition level of 4 gives 2^4 = 16 frequencies)
        """

        self.signal_data = []
        if wave_fam == "db" or wave_fam == "coif":
            if wave_fam == "db" and 1 < wave_mag <= 38:
                self.wavelet = wave_fam + str(wave_mag)
            elif wave_fam == "coif" and 1 <= wave_mag <= 17:
                self.wavelet = wave_fam + str(wave_mag)
            else:
                raise ValueError("Magnitude not compatible.")
        else:
            raise ValueError("Wavelet not compatible.")

        self.mode = "symmetric"
        self.time_bands = time_bands
        self.dec_level = dec_level
        self.wave_coeffs = []

    def load(self, directory: str, n: int):
        """
        Loads compressed data from external file.

        Parameters:
            - directory: directory of data file
            - n: the waveform that is to be decomposed
        """
        x = io.load_compressed_data(directory)
        self.signal_data = x[n, :]
        return self.signal_data

    def load_manual(self, data: list):
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

        if wp.maxlevel < self.dec_level:
            self.dec_level = wp.maxlevel
            warnings.warn(
                "Decomposition level is greater than the maximum allowed level. Consider choosing a lower decomposition level."
            )

        for level in range(1, self.dec_level + 1):
            self.wave_coeffs.append([wp[node.path].data for node in wp.get_level(level, "natural")])

        return self.wave_coeffs

    def decomposer(self) -> Tuple[Dict[str, list], float]:
        """
        Decomposes frequency band energy data into frequency and time band data.

        Parameters:
            - None
        Returns:
            - A dictionary containing the frequency bands and the energies for each time band.
            - The decomposition level energy divided by the total energy.
        """
        energy = []
        time_energy = []
        c = 0
        tot_energy = sum(self.wave_coeffs[0][0][:] ** 2) + sum(self.wave_coeffs[0][1][:] ** 2)

        if len(self.wave_coeffs[self.dec_level - 1][0][:]) < self.time_bands:
            self.time_bands = len(self.wave_coeffs[self.dec_level - 1][0][:])
            warnings.warn(
                "The time band resolution is too high for this decomposition level. Defaulted to maximum allowable resolution (frequency energy is for entire wave). Consider choosing a lower time resolution."
            )

        for i in range(0, 2**self.dec_level):
            for t in range(0, self.time_bands):
                left_t_bound = int(
                    (t / self.time_bands) * len(self.wave_coeffs[self.dec_level - 1][i][:])
                )
                right_t_bound = int(
                    (((t + 1) / self.time_bands) * len(self.wave_coeffs[self.dec_level - 1][i][:]))
                )
                time_energy.append(
                    sum(self.wave_coeffs[(self.dec_level - 1)][i][left_t_bound:right_t_bound] ** 2)
                    / tot_energy
                )
            energy.append(time_energy)
            time_energy = []

        for j in range(0, 2**self.dec_level):
            c = c + sum(energy[j][:])

        energies = {}
        for a in range(0, len(energy)):
            energies[
                f"Frequency Band {a * 2048 / (2**self.dec_level)} - {(a+1)* 2048 / (2**self.dec_level) } kHz"
            ] = energy[a][:]

        return energies, c
