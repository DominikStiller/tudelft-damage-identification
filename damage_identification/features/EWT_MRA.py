import warnings
from typing import Dict, Tuple, Any, Optional

import pywt


class MultiResolutionAnalysis:
    """
    This class extracts the signal data to decompose into energy levels at specified time bands and carrier frequencies.
    """

    def __init__(self, params: Dict[str, Optional[Any]] = None):
        """
        Initializes object with specifications for decomposing signal.

        Parameters:
            - wavelet: the name and magnitude of the wavelet family (e.g.: 'db38' or 'coif17')
            - time_bands: the amount of time bands to split the energy information (i.e. with time_bands = 4, the energy information describes the 4 quarters of the signal)
            - dec_level: the decomposition level of the signal (i.e. the amount of carrier frequencies the signal will be split up into. e.g.: decomposition level of 4 gives 2^4 = 16 frequencies)
        """
        if params is None:
            params = {}
        if "wavelet_decomposition_family" not in params:
            params["wavelet_decomposition_family"] = "db"
        if "wavelet_magnitude" not in params:
            params["wavelet_magnitude"] = 3
        if "decomposition_time_bands" not in params:
            params["decomposition_time_bands"] = 4
        if "decomposition_level" not in params:
            params["decomposition_level"] = 3
        super().__init__()

        if (
            params["wavelet_decomposition_family"] == "db"
            or params["wavelet_decomposition_family"] == "coif"
        ):
            if (
                params["wavelet_decomposition_family"] == "db"
                and 1 <= params["wavelet_magnitude"] <= 38
            ):
                self.wavelet = params["wavelet_decomposition_family"] + str(
                    params["wavelet_magnitude"]
                )
            elif (
                params["wavelet_decomposition_family"] == "coif"
                and 1 <= params["wavelet_magnitude"] <= 17
            ):
                self.wavelet = params["wavelet_decomposition_family"] + str(
                    params["wavelet_magnitude"]
                )
            else:
                raise ValueError("Magnitude not compatible.")
        else:
            raise ValueError("Wavelet not compatible.")

        self.mode = "symmetric"
        self.time_bands = params["decomposition_time_bands"]
        self.dec_level = params["decomposition_level"]

    def _find_wave_coeffs(self, signal_data):
        """
        Decomposes data in object and flattens binary tree into three-dimensional array.

        Parameters:
            - None
        """
        wp = pywt.WaveletPacket(data=signal_data, wavelet=self.wavelet, mode=self.mode)

        if wp.maxlevel < self.dec_level:
            self.dec_level = wp.maxlevel
            warnings.warn(
                "Decomposition level is greater than the maximum allowed level. Consider choosing a lower decomposition level."
            )

        wave_coeffs = []
        for level in range(1, self.dec_level + 1):
            wave_coeffs.append([wp[node.path].data for node in wp.get_level(level, "natural")])

        return wave_coeffs

    def decompose(self, signal_data) -> Tuple[Dict[str, list], float]:
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
        wave_coeffs = self._find_wave_coeffs(signal_data)
        tot_energy = sum(wave_coeffs[0][0][:] ** 2) + sum(wave_coeffs[0][1][:] ** 2)

        if len(wave_coeffs[self.dec_level - 1][0][:]) < self.time_bands:
            self.time_bands = len(wave_coeffs[self.dec_level - 1][0][:])
            warnings.warn(
                "The time band resolution is too high for this decomposition level. Defaulted to maximum allowable resolution (frequency energy is for entire wave). Consider choosing a lower time resolution."
            )

        for i in range(0, 2**self.dec_level):
            for t in range(0, self.time_bands):
                left_t_bound = int(
                    (t / self.time_bands) * len(wave_coeffs[self.dec_level - 1][i][:])
                )
                right_t_bound = int(
                    (((t + 1) / self.time_bands) * len(wave_coeffs[self.dec_level - 1][i][:]))
                )
                time_energy.append(
                    sum(wave_coeffs[(self.dec_level - 1)][i][left_t_bound:right_t_bound] ** 2)
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
