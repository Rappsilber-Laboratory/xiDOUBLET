"""Module containing the base class for all spectrum filters."""
import abc


class BaseFilter(abc.ABC):
    """Base class for spectrum filters."""

    config_needed = True

    def __init__(self, context=None):
        """Initialise the Filter."""
        if self.config_needed and (context is None):
            raise TypeError('Config is required for this filter')
        self.config = getattr(context, 'config', None)
        self.context = context

    @abc.abstractmethod
    def process(self, spectrum):
        """
        Apply the current filter to the given spectrum.

        :param spectrum: (Spectrum) The original spectrum to be processed
        :return: (Spectrum) New processed spectrum
        """
        pass
