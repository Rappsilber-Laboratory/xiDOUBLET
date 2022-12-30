from xidoublet.spectra_reader import Spectrum
from xidoublet.filters import RelativeIntensityFilter, IntensityRankFilter
from numpy.testing import assert_array_equal
import numpy as np

spectrum = Spectrum({'mz': 100, 'charge': 2},
                    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                    [50.0, 1000.0, 49.9999, 51, 800.0, 700.5, 1.1, 5, 20, 100], 'test_scan')
spectrum.charge_values = np.array([1, 2, 1, 1, 2, 3, 3, 2, 3, 4])


def test_rel_intensity_filter():
    # check output is unchanged on 0 cutoff
    out = RelativeIntensityFilter().process(spectrum, 0)
    assert_array_equal(out.mz_values, spectrum.mz_values)
    assert_array_equal(out.int_values, spectrum.int_values)
    assert_array_equal(out.charge_values, spectrum.charge_values)

    # apply 5% cutoff and check output
    out = RelativeIntensityFilter().process(spectrum, 0.05)
    assert_array_equal(out.mz_values, [100, 101, 103, 104, 105, 109])
    assert_array_equal(out.int_values, [50.0, 1000.0, 51, 800.0, 700.5, 100])
    assert_array_equal(out.charge_values, [1, 2, 1, 2, 3, 4])


def test_intensity_rank_filter():
    out = IntensityRankFilter().process(spectrum, 5)
    # top 5 peaks should remain
    assert_array_equal(out.mz_values, [101, 103, 104, 105, 109])
    assert_array_equal(out.int_values, [1000.0, 51, 800.0, 700.5, 100])
    assert_array_equal(out.charge_values, [2, 1, 2, 3, 4])
