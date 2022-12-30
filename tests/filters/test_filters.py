from xidoublet.filters import IsotopeDetector, IsotopeReducer
from xidoublet.detector import SearchContext
from xidoublet.config import Config, Crosslinker


def test_isotope_detector():
    assert IsotopeDetector(SearchContext(Config()))


def test_isotope_reducer():
    assert IsotopeReducer()
