"""Module providing constants."""
import sys


class _const:
    proton_mass = 1.007276466879
    c12c13_massdiff = 1.0033548
    H_MASS = 1.007825032241
    H2O_MASS = 18.0105647

    __file__ = __file__

    class ConstError(TypeError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't rebind const(%s)" % name)
        self.__dict__[name] = value


sys.modules[__name__] = _const()
