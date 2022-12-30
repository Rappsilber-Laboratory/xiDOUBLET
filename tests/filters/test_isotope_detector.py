from xidoublet.spectra_reader import Spectrum
from xidoublet.filters import IsotopeDetector
from xidoublet.config import Config, IsotopeDetectorConfig
from xidoublet import dtypes, const
from xidoublet.detector import SearchContext
import numpy as np
from numpy.testing import assert_array_equal


config = IsotopeDetectorConfig()
ctx = SearchContext(Config(ms2_tol='10ppm', isotope_config=config))


def test_initializer():
    detector = IsotopeDetector(ctx)
    assert detector.config.isotope_config == config


def test_isotope_detector_return_new_spectrum():
    spectrum = Spectrum({'mz': 100, 'charge': 1}, [100, 101.0033548, 102.0067096], [10, 30, 20],
                        'test_scan')
    detector = IsotopeDetector(ctx)
    new_spec = detector.process(spectrum)
    assert new_spec != spectrum
    # check that these arrays actually share the same memory, that
    # pointers point at the same arrays
    assert np.byte_bounds(new_spec.mz_values) == np.byte_bounds(spectrum.mz_values)
    assert np.byte_bounds(new_spec.int_values) == np.byte_bounds(spectrum.int_values)


def test_simple_isotope_detector():
    """Test finding an isotope cluster (charge 1) with 3 exact m/z value peaks."""
    spectrum = Spectrum({'mz': 100, 'charge': 1}, [100, 101.0033548, 102.0067096], [10, 30, 20],
                        'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    assert_array_equal(detected.isotope_cluster_intensity_values, [30])
    assert_array_equal(detected.isotope_cluster_charge_values, [1])
    assert_array_equal(detected.isotope_cluster_mz_values, [100])
    expected_cluster_peaks = np.array(
        [(0, 0), (0, 1), (0, 2)], dtype=dtypes.peak_cluster)
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)


def test_simple_isotope_detector_with_more_peaks():
    """Test finding 2 isotope clusters (charge 1) with 3 exact m/z value peaks each."""
    spectrum = Spectrum({'mz': 100, 'charge': 1},
                        [100, 101.0033548, 102.0067096, 120, 121.0033548, 122.0067096],
                        [100, 30, 10, 110, 31, 11],
                        'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    assert_array_equal(detected.isotope_cluster_mz_values, [100, 120])
    assert_array_equal(detected.isotope_cluster_intensity_values, [100, 110])
    assert_array_equal(detected.isotope_cluster_charge_values, [1, 1])
    expected_cluster_peaks = np.array(
        [(0, 0), (0, 1), (0, 2), (1, 3), (1, 4), (1, 5)],
        dtype=dtypes.peak_cluster)
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)

    # test with summed cluster intensity
    sum_ctx = SearchContext(Config(ms2_tol='10ppm',
                                   isotope_config=IsotopeDetectorConfig(cluster_intensity='sum')))
    detected = IsotopeDetector(sum_ctx).process(spectrum)
    assert_array_equal(detected.isotope_cluster_intensity_values, [140, 152])
    # rest should stay unchanged
    assert_array_equal(detected.isotope_cluster_mz_values, [100, 120])
    assert_array_equal(detected.isotope_cluster_charge_values, [1, 1])
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)


def test_simple_long_cluster():
    """Test finding a very long isotope cluster (charge 1) with 20 exact m/z value peaks."""
    spectrum = Spectrum({'mz': 100, 'charge': 1},
                        [100,
                         100 + 1 * const.c12c13_massdiff,
                         100 + 2 * const.c12c13_massdiff,
                         100 + 3 * const.c12c13_massdiff,
                         100 + 4 * const.c12c13_massdiff,
                         100 + 5 * const.c12c13_massdiff,
                         100 + 6 * const.c12c13_massdiff,
                         100 + 7 * const.c12c13_massdiff,
                         100 + 8 * const.c12c13_massdiff,
                         100 + 9 * const.c12c13_massdiff,
                         100 + 10 * const.c12c13_massdiff,
                         100 + 11 * const.c12c13_massdiff,
                         100 + 12 * const.c12c13_massdiff,
                         100 + 13 * const.c12c13_massdiff,
                         100 + 14 * const.c12c13_massdiff,
                         100 + 15 * const.c12c13_massdiff,
                         100 + 16 * const.c12c13_massdiff,
                         100 + 17 * const.c12c13_massdiff,
                         100 + 18 * const.c12c13_massdiff,
                         100 + 19 * const.c12c13_massdiff],
                        np.full(20, 100),
                        'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    # the first cluster should contain all 20 peaks
    assert_array_equal(detected.isotope_cluster[0].peaks, list(range(len(spectrum.mz_values))))


def test_simple_isotope_detector_with_noise():
    """Test finding an isotope cluster (charge 1) with 5 exact peaks + noise peak in between."""
    spectrum = Spectrum({'mz': 100, 'charge': 1},
                        [100, 101.0033548, 102.0067096, 102.5, 103.0100644, 104.0134192],
                        [100, 4, 0.002, 110, 0.0005, 0.0002],
                        'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    assert_array_equal(detected.isotope_cluster_mz_values, [100, 102.5])
    assert_array_equal(detected.isotope_cluster_intensity_values, [100, 110])
    assert_array_equal(detected.isotope_cluster_charge_values, [1, 0])
    expected_cluster_peaks = np.array(
        [(0, 0), (0, 1), (0, 2), (0, 4), (0, 5), (1, 3)],
        dtype=dtypes.peak_cluster)
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)


def test_simple_isotope_detector_with_more_peaks_and_noise():
    """Test finding 2 isotope clusters with multiple noise peaks in between."""
    spectrum = Spectrum(
        {'mz': 100, 'charge': 1},
        [100, 100.1, 100.5, 101.0033548, 102.0067096, 120, 121.0033548, 121.1, 122.0067096],
        [100, 2, 3, 30, 10, 110, 31, 8, 11],
        'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    assert_array_equal(detected.isotope_cluster_mz_values, [100, 100.1, 100.5, 120, 121.1])
    assert_array_equal(detected.isotope_cluster_intensity_values, [100, 2, 3, 110, 8])
    assert_array_equal(detected.isotope_cluster_charge_values, [1, 0, 0, 1, 0])
    expected_cluster_peaks = np.array(
        [(0, 0), (0, 3), (0, 4), (1, 1), (2, 2), (3, 5), (3, 6), (3, 8), (4, 7)],
        dtype=dtypes.peak_cluster)
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)


def test_simple_isotope_detector_with_inexact_peaks():
    """Test finding 2 isotope clusters with inexact m/z values + noise."""
    spectrum = Spectrum(
        {'mz': 100, 'charge': 1},
        [100, 100.1, 100.5, 101.004, 102.006, 120, 121.0041, 121.1, 122.007],
        [100, 2, 3, 30, 10, 110, 31, 8, 11],
        'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    assert_array_equal(detected.isotope_cluster_mz_values, [100, 100.1, 100.5, 120, 121.1])
    assert_array_equal(detected.isotope_cluster_intensity_values, [100, 2, 3, 110, 8])
    assert_array_equal(detected.isotope_cluster_charge_values, [1, 0, 0, 1, 0])
    expected_cluster_peaks = np.array(
        [(0, 0), (0, 3), (0, 4), (1, 1), (2, 2), (3, 5), (3, 6), (3, 8), (4, 7)],
        dtype=dtypes.peak_cluster)
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)


def test_simple_isotope_detector_with_tolerance():
    """Test finding a single isotope cluster (charge 1) with error tolerance."""
    spectrum = Spectrum({'mz': 100, 'charge': 1}, [100, 101.1, 102.0], [10, 30, 20], 'test_scan')
    config = IsotopeDetectorConfig(rtol=1e-3)
    detected = IsotopeDetector(SearchContext(Config(isotope_config=config))).process(spectrum)
    assert_array_equal(detected.isotope_cluster_mz_values, [100])
    assert_array_equal(detected.isotope_cluster_intensity_values, [30])
    assert_array_equal(detected.isotope_cluster_charge_values, [1])
    expected_cluster_peaks = np.array(
        [(0, 0), (0, 1), (0, 2)], dtype=dtypes.peak_cluster)
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)


def test_isotope_detection_with_charge():
    """Test finding an isotope clusters in charge state 2 (and 1)."""
    spectrum = Spectrum({'mz': 100, 'charge': 2},
                        [100, 100.5016, 101.0033, 102.0067],
                        [50, 30, 10, 10],
                        'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    assert_array_equal(detected.isotope_cluster_mz_values, [100, 100])
    assert_array_equal(detected.isotope_cluster_intensity_values, [50, 50])
    assert_array_equal(detected.isotope_cluster_charge_values, [2, 1])
    expected_cluster_peaks = np.array(
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (1, 3)],
        dtype=dtypes.peak_cluster)
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)


def test_multiple_candidates_for_isotope():
    """Test selecting the best matching of multiple peaks matching an isotopic peak."""
    spectrum = Spectrum({'mz': 100, 'charge': 1},
                        [100, 101.0033548, 101.0034],
                        [100, 30, 10],
                        'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    # 101.0033548 & 101.0034 both match the first isotopic peak,
    # but 101.0033548 has the smaller error so 101.0034 is not included in the cluster
    assert_array_equal(detected.isotope_cluster_mz_values, [100, 101.0034])
    assert_array_equal(detected.isotope_cluster_intensity_values, [100, 10])
    assert_array_equal(detected.isotope_cluster_charge_values, [1, 0])
    expected_cluster_peaks = np.array(
        [(0, 0), (0, 1), (1, 2)], dtype=dtypes.peak_cluster)
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)


def test_isotope_detection_with_different_charges():
    """Test finding isotope clusters in charge states 1 & 2 with noise peaks."""
    spectrum = Spectrum(
        {'mz': 100, 'charge': 2},
        [100, 100.1, 100.7, 101.004, 102.006,
         120, 120.5016, 121.0033, 121.5050],
        [100, 2, 3, 30, 10,
         110, 31, 8, 11],
        'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    assert_array_equal(detected.isotope_cluster_mz_values, [100, 100.1, 100.7, 120, 120])
    assert_array_equal(detected.isotope_cluster_intensity_values, [100, 2, 3, 110, 110])
    assert_array_equal(detected.isotope_cluster_charge_values, [1, 0, 0, 1, 2])
    expected_cluster_peaks = np.array(
        [(0, 0), (0, 3), (0, 4), (1, 1), (2, 2), (3, 5), (3, 7), (4, 5), (4, 6), (4, 7), (4, 8)],
        dtype=dtypes.peak_cluster)
    assert_array_equal(detected.isotope_cluster_peaks, expected_cluster_peaks)


def test_filter_cluster():
    """Test filtering/splitting clusters that are fully/partially contained in other clusters."""
    detector = IsotopeDetector(ctx)
    # peak-mz - does not really matter for this test - as everything is just by id
    peak_mz = np.array([100, 100.25, 100.5, 101, 102, 120, 120.5, 121, 121.5])
    peak_int = np.full(peak_mz.shape, 100)

    spectrum = Spectrum(
        {'mz': 100, 'charge': 2}, peak_mz, peak_int,
        'test_scan')
    spectrum.peak_has_cluster = np.zeros(len(spectrum.mz_values), dtype=bool)

    clusters = [
        # 0
        IsotopeDetector.Cluster(spectrum, charge=1, peaks=np.array([0, 2, 3, 4])),
        # 1 - complete overlap with 0 - should disappear
        IsotopeDetector.Cluster(spectrum, charge=1, peaks=np.array([2, 3, 4])),
        # 2 - same start as 0 - should stay
        IsotopeDetector.Cluster(spectrum, charge=4, peaks=np.array([0, 1, 2])),
        # 3 - same start as 0 - should stay
        IsotopeDetector.Cluster(spectrum, charge=2, peaks=np.array([0, 2, 3])),
        # 4 - fully contained in prior cluster - should go
        IsotopeDetector.Cluster(spectrum, charge=1, peaks=np.array([1, 2, 3])),
        # 5 - overlap of first but not second peak
        #    should come back on it's own and as a cluster [5, 6, 7] (first non-overlap)
        IsotopeDetector.Cluster(spectrum, charge=1, peaks=np.array([4, 5, 6, 7])),
    ]

    expected_cluster = [
        clusters[0],
        clusters[2],
        clusters[3],
        clusters[5],
        IsotopeDetector.Cluster(spectrum, charge=1, peaks=np.array([5, 6, 7]))
    ]

    ret_cluster = detector.filter_cluster(clusters, spectrum)
    for x, y in zip(ret_cluster, expected_cluster):
        if x != y:
            # no need to compare untouched cluster
            assert_array_equal(x.peaks, y.peaks)
            assert x.charge == y.charge


def test_relative_height():
    """very limited test - that just makes sure we have the expected top-peak for different
    masses"""
    detector = IsotopeDetector(ctx)
    # first peak should be the largest
    r = detector.relative_height(100)
    assert r[0] == np.max(r)
    r = detector.relative_height(2000)
    # for 2000 the second peak should be largest
    assert r[0] < np.max(r)
    assert r[1] == np.max(r)
    assert r[2] < np.max(r)

    r = detector.relative_height(20000)
    # for 20000 the cluster should have a caculated mass for all peaks
    assert r[-1] < r[-2]


def test_large_cluster():
    masses = np.array(
        [100, 101.0033548, 102.0067096, 103.0100644, 104.0134192,
         105.016774, 106.0201288, 107.0234836, 108.0268384]
    )
    spectrum = Spectrum({'mz': 100, 'charge': 1}, masses, np.full(9, 100), 'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    # we should have several cluster - but one should be over the full length
    assert len([c for c in detected.isotope_cluster if c.length == 9]) == 1


def test_ignore_small_leading_peak():
    masses = np.array(
        [100, 101.0033548, 102.0067096, 103.0100644]
    )
    intensities = np.array(
        [1, 100, 5, 1]
    )
    spectrum = Spectrum({'mz': 100, 'charge': 1}, masses, intensities, 'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    # we should have one cluster starting at peak 1 and a single peak "cluster" at 0
    assert len(detected.isotope_cluster) == 2
    assert_array_equal(detected.isotope_cluster[0].peaks, [0])
    assert_array_equal(detected.isotope_cluster[1].peaks, [1, 2, 3])


def test_overlapping_cluster():
    masses = np.array(
        [100, 101.0033548, 102.0067096, 103.0100644, 104.0134192,  # cluster charge 1
         105.016774,  # this peak matches to above cluster charge 1 but also to a charge 2 cluster
         105.5184514, 106.0201288, 106.521806, 107.0234836, 107.525161,
         120]  # single peak should have it's own cluster
    )
    intensities = np.array(
        [100, 4, 0.002, 0.0005, 0.0002,  # cluster charge 1
         100,  # this peak matches to above cluster charge 1 but also to a charge 2 cluster
         100, 10, 0.6, 5000, 400,
         120]  # single peak should have it's own cluster
    )
    spectrum = Spectrum({'mz': 100, 'charge': 2}, masses, intensities, 'test_scan')
    detected = IsotopeDetector(ctx).process(spectrum)
    assert_array_equal(detected.isotope_cluster_peaks, np.array(
                       [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 7), (0, 9),  # z1,mz100
                        (1, 5), (1, 7), (1, 9),  # z1, mz105.016
                        (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),  # z2,mz105.016
                        (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),  # z2,mz105.5
                        (4, 9), (4, 10),  # z2, mz 107.02 - split by averagene
                        (5, 11),  # z0, mz 120
                        ], dtype=detected.isotope_cluster_peaks.dtype))

    assert_array_equal(detected.isotope_cluster_charge_values, [1, 1, 2, 2, 2, 0])
    assert_array_equal(detected.isotope_cluster_mz_values, [masses[0], masses[5], masses[5],
                                                            masses[6], masses[9], masses[11]])
