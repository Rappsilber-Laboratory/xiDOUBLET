from xidoublet.doublet import *
from xidoublet.spectra_reader import Spectrum
from xidoublet.config import Config
from xidoublet.detector import SearchContext
from xidoublet import const
import numpy as np
from numpy.testing import assert_array_almost_equal
import os

fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')


def test_find_doublets():
    """Test finding doublets in a test spectrum."""
    # setup config and context
    config = Config(ms2_tol='15ppm', crosslinker={
        'name': 'test',
        'mass': 200,
        'cleavage_stubs': [{'name': 'a', 'mass': 42},
                           {'name': 'b', 'mass': 142}]
    }, doublet={'undefined_charge_match': False, 'stubs': ['a', 'b']})
    ctx = SearchContext(config)

    spectrum = Spectrum({'mz': 200 + const.proton_mass, 'charge': 3},
                        [100, 140, 150, 200, 300, 400.004],
                        [9.0, 10.0, 10.0, 4.0, 1.5, 7.3],
                        'test_scan')
    spectrum.charge_values = np.array([2, 2, 2, 0, 1, 1])

    cl_mass = config.crosslinker.mass
    expected_doublets = np.array([
        # p0_mz, p1_mz, p0_rel_int, p1_rel_int, p0_rank, p1_rank, charge, pep_mass, 2nd_pep_mass,
        # abs_error, rel_error
        (100, 150, 0.9, 1.0, 2, 1, 2, 155.985447, 600-(155.985447+cl_mass), 0, 0),
        (300, 400.004, 0.15, 0.73, 5, 3, 1, 256.992723, 600-(256.992723+cl_mass), 0.004, 1e-5),
    ], dtype=dtypes.doublet_dtype)

    b_stub_mass = 42
    c_stub_mass = 142

    doublets = find_doublets(spectrum, b_stub_mass, c_stub_mass, ctx)

    for name in expected_doublets.dtype.names:
        assert_array_almost_equal(doublets[name], expected_doublets[name])


def test_find_doublets_undef_charge():
    """Test finding doublets in a test spectrum with undefined charge state peaks."""
    # setup config and context
    config = Config(ms2_tol='15ppm', crosslinker={
        'name': 'test',
        'mass': 200,
        'cleavage_stubs': [{'name': 'a', 'mass': 42},
                           {'name': 'b', 'mass': 142}]
    }, doublet={'undefined_charge_match': True, 'stubs': ['a', 'b']})
    ctx = SearchContext(config)

    spectrum = Spectrum({'mz': 200 + const.proton_mass, 'charge': 3},
                        [100, 140, 150, 200, 300, 400.004],
                        [9.0, 10.0, 10.0, 4.0, 1.5, 7.3],
                        'test_scan')
    spectrum.charge_values = np.array([2, 2, 2, 0, 1, 1])

    cl_mass = config.crosslinker.mass
    expected_doublets = np.array([
        # p0_mz, p1_mz, p0_rel_int, p1_rel_int, p0_rank, p1_rank, charge, pep_mass, 2nd_pep_mass,
        # abs_error, rel_error
        (100, 150, 0.9, 1.0, 2, 1, 2, 155.985447, 600 - (155.985447 + cl_mass), 0, 0),
        (150, 200, 1.0, 0.4, 1, 4, 2, 255.98544707, 600 - (255.98544707 + cl_mass), 0, 0),
        (200, 300, 0.4, 0.15, 4, 5, 1, 156.99272353, 600 - (156.99272353 + cl_mass), 0, 0),
        (300, 400.004, 0.15, 0.73, 5, 3, 1, 256.992723, 600 - (256.992723 + cl_mass), 0.004, 1e-5),
    ], dtype=dtypes.doublet_dtype)

    b_stub_mass = 42
    c_stub_mass = 142

    doublets = find_doublets(spectrum, b_stub_mass, c_stub_mass, ctx)

    for name in expected_doublets.dtype.names:
        assert_array_almost_equal(doublets[name], expected_doublets[name])
