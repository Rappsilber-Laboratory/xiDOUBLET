from xidoublet.filters.base_filter import BaseFilter
import pytest


class BadFilter(BaseFilter):
    pass


class GoodFilter(BaseFilter):
    def process(self, spectrum):
        pass


def test_base_filter_is_abc():
    with pytest.raises(TypeError):
        BaseFilter('x')


def test_child_class_fails_when_no_process_method_is_defined():
    with pytest.raises(TypeError):
        BadFilter('x')


def test_base_filter_has_abstract_method_process():
    assert GoodFilter('x')
