import numpy as np
from abc import ABC, abstractmethod
import os


class ResultWriter(ABC):
    """ Class to write results. """

    @abstractmethod
    def write(self, result):
        """
        Write result.

        :param result: (np.array) result to write.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the writer.
        """
        pass

    def put(self, result):
        self.write(result)


class TsvResultWriter(ResultWriter):
    """Write results as tab-separated values."""

    def __init__(self, out_path):
        """
        Initialize the tsv result writer.

        :param out_path: (str) where to write the result file.
        """
        if os.path.isfile(out_path) and os.stat(out_path).st_size != 0:
            self._header_written = True
        else:
            self._header_written = False
        self.file = open(out_path, 'a')

    def write(self, result):
        """
        Append result to tsv file. writes a header if it's the first result for this file.

        :param result: (np.array) result to write.
        """
        if not self._header_written:
            header = "\t".join(result.dtype.names)
            np.savetxt(self.file, result, delimiter='\t', fmt="%s", header=header, comments='')
            self._header_written = True
        else:
            np.savetxt(self.file, result, delimiter='\t', fmt="%s")

    def close(self):
        self.file.close()


class MemoryResultWriter(ResultWriter):
    """Write results to a list in memory."""

    def __init__(self):
        """Initialise the memory result writer."""
        self._results = []

    def write(self, result):
        """
        Append result to out_list.

        :param result: (np.array) result to write.
        """
        self._results.append(result)

    def close(self):
        """Close not needed when writing to memory."""
        pass

    def results(self):
        """Return results."""
        return self._results
