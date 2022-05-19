from typing import Any, Optional

import numpy as np
import pywt

from damage_identification.features.base import FeatureExtractor


class MultiResolutionAnalysisExtractor(FeatureExtractor):
    """
    This class extracts the signal data to decompose into energy levels at specified time bands and carrier frequencies.
    """

    def __init__(self, params: dict[str, Optional[Any]] = None):
        """
        Initializes object with specifications for decomposing signal.

        Parameters:
            - wavelet: the name and magnitude of the wavelet family (e.g.: 'db38' or 'coif17')
            - time_bands: the amount of time bands to split the energy information (i.e. with time_bands = 4, the energy information describes the 4 quarters of the signal)
            - dec_level: the decomposition level of the signal (i.e. the amount of carrier frequencies the signal will be split up into. e.g.: decomposition level of 4 gives 2^4 = 16 frequencies)
        """
        if params is None:
            params = {}
        if "mra_wavelet_family" not in params:
            params["mra_wavelet_family"] = "db"
        if "mra_wavelet_scale" not in params:
            params["mra_wavelet_scale"] = 3
        if "mra_time_bands" not in params:
            params["mra_time_bands"] = 4
        if "mra_levels" not in params:
            params["mra_levels"] = 3

        if params["mra_wavelet_family"] == "db" or params["mra_wavelet_family"] == "coif":
            if params["mra_wavelet_family"] == "db" and 1 <= params["mra_wavelet_scale"] <= 38:
                self.wavelet = params["mra_wavelet_family"] + str(params["mra_wavelet_scale"])
            elif params["mra_wavelet_family"] == "coif" and 1 <= params["mra_wavelet_scale"] <= 17:
                self.wavelet = params["mra_wavelet_family"] + str(params["mra_wavelet_scale"])
            else:
                raise ValueError("Magnitude not compatible.")
        else:
            raise ValueError("Wavelet not compatible.")

        self.mode = "symmetric"
        self.time_bands = params["mra_time_bands"]
        self.dec_level = params["mra_levels"]
        super().__init__("mra", params)

    def extract_features(self, example: np.ndarray) -> dict[str, float]:
        """
        Decomposes frequency band energy data into frequency and time band data.

        Returns:
            A dictionary containing the frequency bands and the energies for each time band.
        """
        wave_coeffs, dec_level = self._find_wavelet_coeffs(example)
        total_energy = sum(wave_coeffs[0][0][:] ** 2) + sum(wave_coeffs[0][1][:] ** 2)

        # Determine number of time bands
        max_time_bands = len(wave_coeffs[dec_level - 1][0][:])
        time_bands = self.time_bands
        if max_time_bands < time_bands:
            time_bands = max_time_bands
            print(
                "WARNING: The time band resolution is too high for this decomposition level. "
                "Defaulted to maximum allowable resolution (frequency energy is for entire wave). "
                "Consider choosing a lower time resolution."
            )

        if total_energy != 0:
            energies = []  # will be a list of lists, first decomposition levels, then time bands
            # Calculate band energies
            for i in range(0, 2**dec_level):
                time_energy = []
                for t in range(0, time_bands):
                    left_t_bound = int((t / time_bands) * len(wave_coeffs[dec_level - 1][i][:]))
                    right_t_bound = int(
                        (((t + 1) / time_bands) * len(wave_coeffs[dec_level - 1][i][:]))
                    )
                    time_energy.append(
                        sum(wave_coeffs[(dec_level - 1)][i][left_t_bound:right_t_bound] ** 2)
                        / total_energy
                    )

                energies.append(time_energy)
        else:
            # Mark example as invalid if total energy is zero
            energies = [[None] * time_bands] * 2**dec_level

        # Assemble feature dictionary
        features = {}

        for level, energies_per_level in enumerate(energies):
            band_lower = level * 2048 / (2**dec_level)
            band_upper = (level + 1) * 2048 / (2**dec_level)
            if level < 4:  # only considers the frequencies until 1024 kHz
                for band, band_energy in enumerate(energies_per_level):
                    features[f"mra_{band_lower:.0f}_{band_upper:.0f}_{band}"] = band_energy

        return features

    def _find_wavelet_coeffs(self, signal_data: np.ndarray) -> tuple[list[list[float]], int]:
        """
        Decomposes data in object and flattens binary tree into three-dimensional array.
        """
        # Perform wavelet packet transform
        wp = pywt.WaveletPacket(data=signal_data, wavelet=self.wavelet, mode=self.mode)

        dec_level = self.dec_level
        if wp.maxlevel < dec_level:
            dec_level = wp.maxlevel
            print(
                "WARNING: Decomposition level is greater than the maximum allowed level. Consider choosing a lower decomposition level."
            )

        # Extract wavelet coefficients
        wavelet_coeffs = []
        for level in range(1, dec_level + 1):
            wavelet_coeffs.append([wp[node.path].data for node in wp.get_level(level, "natural")])

        return wavelet_coeffs, dec_level
