from xidoublet.detector import *
from xidoublet.config import Config, Crosslinker, Stub
import os
from xidoublet.spectra_reader import PeakListWrapper
from xidoublet.result_writer import MemoryResultWriter


test_cl = Crosslinker(name='test',
                      mass=100.1,
                      cleavage_stubs=[
                          Stub(name='a', mass=10.5),
                          Stub(name='b', mass=42.0),
                      ])


def test_context():
    config = Config(crosslinker=test_cl, doublet={"stubs": ["a", "b"]})
    context = SearchContext(config)
    expected_stub_comb = {'a_b': (10.5, 42.0)}
    assert context.stub_combinations == expected_stub_comb

    # check that swapping the stub order doesn't make a difference
    config = Config(crosslinker=test_cl, doublet={"stubs": ["b", "a"]})
    context = SearchContext(config)
    assert context.stub_combinations == expected_stub_comb


def test_processor():
    cfg = Config(crosslinker=Crosslinker.DSSO, doublet={
        "stubs": ["a", "t"],
        "cap": 4,
        "second_peptide_mass_filter": 500,
        "mz_window_filter": 1.5
    })
    ctx = SearchContext(cfg)

    # init
    processor = SpectrumProcessor(ctx)
    spectra_wrapper = PeakListWrapper(ctx)
    doublet_writer = MemoryResultWriter()
    pair_writer = MemoryResultWriter()

    # load
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    peaklist_file = os.path.join(fixtures_dir, 'test_processor.mgf')
    spectra_wrapper.load(peaklist_file)

    # process
    for spectrum in spectra_wrapper.spectra:
        processor.process_spectrum(spectrum, doublet_writer)
    doublet_writer.close()

    doublet_results = doublet_writer.results()

    assert len(doublet_results[0]) == 4
