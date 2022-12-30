from copy import copy
from xidoublet.filters.base_filter import BaseFilter


class IsotopeReducer(BaseFilter):
    """Filter that reduces all detected isotope clusters to their monoisotopic peaks."""

    config_needed = False

    def process(self, spectrum):
        """
        Process a spectrum, removing all but the monoisotopic peak for detected isotope clusters.

        Input spectrum needs to have annotated isotope clusters (`isotope_cluster_ids`).
        Intensities of all other cluster peaks get added to the monoisotopic peak.
        :param spectrum: (Spectrum) Isotope detected spectrum to process
        :return: (Spectrum) Isotope reduced copy of the spectrum
        """
        # check if input spectrum is isotope detected
        if spectrum.isotope_cluster_charge_values is None:
            raise ValueError
        new_spec = copy(spectrum)

        new_spec.int_values = spectrum.isotope_cluster_intensity_values
        new_spec.mz_values = spectrum.isotope_cluster_mz_values
        new_spec.charge_values = spectrum.isotope_cluster_charge_values

        new_spec.isotope_cluster_mz_values = None
        new_spec.isotope_cluster_intensity_values = None
        new_spec.isotope_cluster_charge_values = None
        new_spec.isotope_cluster_peaks = None
        new_spec.peak_has_cluster = None
        new_spec.isotope_cluster = None

        return new_spec
