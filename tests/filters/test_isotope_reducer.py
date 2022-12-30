from xidoublet.spectra_reader import Spectrum
from xidoublet.filters import IsotopeReducer
from numpy.testing import assert_array_equal
import pytest
import numpy as np


def test_raise_on_no_charge_data():
    spectrum = Spectrum({'mz': 100, 'charge': 2}, [100, 101], [1000, 20], 'test_scan')
    with pytest.raises(ValueError):
        IsotopeReducer().process(spectrum)


def test_trivial_case():
    spectrum = Spectrum({'mz': 100, 'charge': 2}, [100, 101], [1000, 20], 'test_scan')
    spectrum.isotope_cluster_ids = np.array([0, 1])
    spectrum.isotope_cluster_charge_values = np.array([0, 0])
    spectrum.isotope_cluster_mz_values = spectrum.mz_values
    spectrum.isotope_cluster_intensity_values = spectrum.int_values
    out = IsotopeReducer().process(spectrum)
    assert_array_equal(out.mz_values, [100, 101])
    assert_array_equal(out.int_values, [1000, 20])


def test_isotope_reducer():
    spectrum = Spectrum({'mz': 100, 'charge': 2},
                        [90, 91, 92, 98, 100, 101, 102, 103],
                        [2000, 200, 20, 500, 1000, 20, 10, 5], 'test_scan')
    spectrum.isotope_cluster_charge_values = np.array([1, 2, 3])
    spectrum.isotope_cluster_mz_values = np.array([90, 98, 100])
    spectrum.isotope_cluster_intensity_values = np.array([2000, 500, 1000])
    out = IsotopeReducer().process(spectrum)
    assert_array_equal(out.mz_values, [90, 98, 100])
    assert_array_equal(out.int_values, [2000, 500, 1000])
    assert_array_equal(out.charge_values, [1, 2, 3])
