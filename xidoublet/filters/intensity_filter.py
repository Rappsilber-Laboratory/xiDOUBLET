from xidoublet.filters.base_filter import BaseFilter
from copy import copy
import numpy as np


class RelativeIntensityFilter(BaseFilter):
    """
    Applies an intensity cutoff relative to the base peak of the spectrum.
    """
    config_needed = False

    def process(self, spectrum, cut_off=0.05):
        """
        Apply a relative intensity based cutoff on a spectrum.

        :param spectrum: (Spectrum) Spectrum to process
        :param cut_off: (float) Relative intensity threshold
        :return: (Spectrum) new Spectrum
        """
        int_values = spectrum.int_values

        int_mask = int_values / int_values.max() >= cut_off

        new_spec = copy(spectrum)

        new_spec.mz_values = spectrum.mz_values[int_mask]
        new_spec.int_values = int_values[int_mask]
        if hasattr(new_spec, 'charge_values') and new_spec.charge_values is not None:
            new_spec.charge_values = spectrum.charge_values[int_mask]

        return new_spec


class IntensityRankFilter(BaseFilter):
    """
    Apply an intensity rank based cutoff on the spectrum.
    """
    config_needed = False

    def process(self, spectrum, min_rank=20):
        """
        Apply a intensity rank based cutoff on a spectrum.

        :param spectrum: (Spectrum) Spectrum to process
        :param min_rank: (int) Lowest intensity rank to keep
        :return: (Spectrum) new Spectrum
        """
        int_values = spectrum.int_values

        # create new spectrum
        new_spec = copy(spectrum)

        if int_values.size <= min_rank:
            return new_spec

        # get the indices of the top n peaks
        top_idx = np.argpartition(-int_values, min_rank)[:min_rank]

        # filter mz values and reorder
        new_spec.mz_values = spectrum.mz_values[top_idx]
        mz_ordered = np.argsort(new_spec.mz_values)
        new_spec.mz_values = new_spec.mz_values[mz_ordered]

        # filter and order remaining arrays
        new_spec.int_values = int_values[top_idx][mz_ordered]
        if hasattr(new_spec, 'charge_values') and new_spec.charge_values is not None:
            new_spec.charge_values = \
                spectrum.charge_values[top_idx][mz_ordered]

        return new_spec
