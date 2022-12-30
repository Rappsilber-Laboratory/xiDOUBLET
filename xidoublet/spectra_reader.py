from pyteomics import mgf, mzml
from xidoublet import const
import numpy as np
import re
import ntpath
import mmap
from abc import ABC, abstractmethod
import zipfile
import tarfile
import io
import os
from lxml.etree import XMLSyntaxError


class Spectrum:
    def __init__(self, precursor, mz_array, int_array, scan_id, rt=np.nan, file_name='',
                 source_path='', run_name='', scan_number=-1, scan_index=-1, title=''):
        """
        Initialise a Spectrum object.

        :param precursor: (dict) Spectrum precursor information as dict.  e.g. {'mz':
            102.234, 'charge': 2, 'intensity': 12654.35}
        :param mz_array: (ndarray, dtype: float64) m/z values of the spectrum peaks
        :param int_array: (ndarray, dtype: float64) intensity values of the spectrum peaks
        :param scan_id: (str) Unique scan identifier
        :param rt: (str) Retention time in seconds (can be a range, e.g 60-62)
        :param file_name: (str) Name of the peaklist file
        :param source_path: (str) Path to the peaklist source
        :param run_name: (str) Name of the MS run
        :param scan_number: (int) Scan number of the spectrum
        :param scan_index: (int) Index of the spectrum in the file
        """
        self.precursor = precursor
        self.scan_id = scan_id
        self.scan_number = scan_number
        self.scan_index = scan_index
        self.rt = rt
        self.file_name = file_name
        self.source_path = source_path
        self.run_name = run_name
        self.title = title
        mz_array = np.asarray(mz_array, dtype=np.float64)
        int_array = np.asarray(int_array, dtype=np.float64)
        # make sure that the m/z values are sorted asc
        sorted_indices = np.argsort(mz_array)
        self.mz_values = mz_array[sorted_indices]
        self.int_values = int_array[sorted_indices]
        self.peak_has_cluster = None
        self._precursor_mass = None
        self.isotope_cluster = None
        self.isotope_cluster_peaks = None
        self.isotope_cluster_mz_values = None
        self.isotope_cluster_intensity_values = None
        self.isotope_cluster_charge_values = None

    @property
    def precursor_charge(self):
        """Get the precursor charge state."""
        return self.precursor['charge']

    @property
    def precursor_mz(self):
        """Get the precursor m/z."""
        return self.precursor['mz']

    @precursor_mz.setter
    def precursor_mz(self, mz):
        self._precursor_mass = None
        self.precursor['mz'] = mz

    @property
    def precursor_int(self):
        """Get the precursor intensity."""
        return self.precursor['intensity']

    @property
    def precursor_mass(self):
        """Return the neutral mass of the precursor."""
        if self._precursor_mass is None:
            self._precursor_mass = (self.precursor['mz'] - const.proton_mass) *\
                self.precursor['charge']
        return self._precursor_mass

    def initialise_isotope_clusters(self):
        self.peak_has_cluster = np.zeros(len(self.mz_values), dtype=bool)


class PeakListWrapper:
    """Wrapper holding SpectraReaders."""

    def __init__(self, context, offset=0, step=1):
        """
        Initialise the PeakListWrapper.

        :param context: (Searcher) Search context
        """
        self.readers = None
        self.context = context
        self.offset = offset
        self.step = step

    def load(self, peaklist_files, reset=True):
        """
        Create SpectraReaders from peaklist files.

        Supported file types: MGF and mzML (and tar or zip archives)
        :param peaklist_files: (str | list of str) path(s) to the peak list file(s) or archive(s)
        """
        if type(peaklist_files) != list:
            peaklist_files = [peaklist_files]

        if reset:
            self.readers = []
        for peaklist_file in peaklist_files:
            count_readers = len(self.readers)
            if not os.path.exists(peaklist_file):
                raise ValueError(f"{peaklist_file} does not exist")
            # check if it is a directory
            if os.path.isdir(peaklist_file):
                # try to load any file in that directory
                for filename in os.listdir(peaklist_file):
                    self.load(os.path.join(peaklist_file, filename), reset=False)
            # check for zipfile
            elif zipfile.is_zipfile(peaklist_file):
                zip_f = zipfile.ZipFile(peaklist_file)
                for member in zip_f.infolist():
                    self._load(zip_f.open(member), member.filename, peaklist_file)
            # check for tarfile
            elif tarfile.is_tarfile(peaklist_file):
                tar_f = tarfile.open(peaklist_file)
                for member in tar_f.getmembers():
                    self._load(tar_f.extractfile(member), member.name)
            # else load file
            else:
                self._load(peaklist_file, peaklist_file)
            if reset:
                if count_readers == len(self.readers):
                    raise ValueError(f"{peaklist_file} could not be loaded")

    def _load(self, stream, filename, source_path=None):
        """Check the file type of the stream and load it."""
        filename = ntpath.basename(filename)
        if filename.lower().endswith('.mgf'):
            if type(stream) == str:
                stream = open(stream)
            else:
                stream = io.TextIOWrapper(stream)
            self.readers.append(MGFReader(self.context))
        elif filename.lower().endswith('.mzml'):
            self.readers.append(MZMLReader(self.context))
        else:
            return
        self.readers[-1].load(stream, source_path=source_path,
                              file_name=filename, step=self.step, offset=self.offset)

    def count_spectra(self):
        """
        Count the number of spectra.

        :return (int) Total number of spectra in all files.
        """
        return sum([r.count_spectra() for r in self.readers])

    @property
    def spectra(self):
        for reader in self.readers:
            for spectrum in reader.spectra:
                yield spectrum


class SpectraReader(ABC):
    """Abstract Base Class for all SpectraReader."""

    def __init__(self, context):
        """
        Initialize the SpectraReader.

        :param context: (Searcher) Search context
        """
        config = context.config
        self._reader = None
        self._re_scan_number = re.compile(config.re_scan_number)
        self._re_run_name = re.compile(config.re_run_name)
        self. _source = None
        self.file_name = None
        self.source_path = None
        self.default_run_name = self.file_name
        self._indexed = False

    @abstractmethod
    def load(self, source, file_name=None, source_path=None):
        """
        Load the spectrum file.

        :param source: Spectra file source
        :param file_name: (str) filename
        :param source_path: (str) path to the source file (peak list file or archive)
        """
        self._source = source
        if source_path is None:
            if type(source) == str:
                self.source_path = source
            elif issubclass(type(source), io.TextIOBase) or \
                    issubclass(type(source), tarfile.ExFileObject):
                self.source_path = source.name
        else:
            self.source_path = source_path

        if file_name is None:
            self.file_name = ntpath.basename(self.source_path)
        else:
            self.file_name = file_name

    @abstractmethod
    def count_spectra(self):
        """
        Count the number of spectra.

        :return (int) Number of spectra in the file
        """
        ...

    @property
    @abstractmethod
    def spectra(self):
        """Create a Spectra generator."""
        while False:
            yield None

    def __getitem__(self, _):
        raise TypeError("Indexed spectra access is not possible for this class.")


class MGFReader(SpectraReader):
    """SpectraReader for MGF files."""

    def load(self, source, file_name=None, source_path=None, offset=0, step=1, indexed=False):
        """
        Load MGF file.

        :param source: file source, path or stream
        :param file_name: (str) MGF filename
        :param source_path: (str) path to the source file (MGF or archive)
        :param indexed: (bool) Create an IndexedMGF for index access of spectra
        """
        self._reader = mgf.read(source, use_index=indexed)
        self._indexed = indexed
        self._reader = mgf.read(source, use_index=False)
        self.offset = offset
        self.step = step
        super().load(source, file_name, source_path)

    def count_spectra(self):
        """
        Count the number of spectra.

        :return (int) Number of spectra in the file.
        """
        if issubclass(type(self._source), io.TextIOBase):
            text = self._source.read()
            result = len(list(re.findall('BEGIN IONS', text)))
            self._source.seek(0)
        else:
            with open(self._source, 'r+') as f:
                text = mmap.mmap(f.fileno(), 0)
                result = len(list(re.findall(b'BEGIN IONS', text)))
                text.close()
        count_from_offset = result-self.offset
        return int(count_from_offset/self.step) + (count_from_offset % self.step > 0)

    def _convert_spectrum(self, scan_index, mgf_spec):
        precursor = {
            'mz': mgf_spec['params']['pepmass'][0],
            'charge': mgf_spec['params']['charge'][0],
            'intensity': mgf_spec['params']['pepmass'][1]
        }

        # use title as scan_id, default to filename_scan_index (very unlikely to not have a
        # title but it's not required)
        scan_id = mgf_spec['params'].get('title', '{}_{}'.format(self.file_name, scan_index))

        # parse retention time, default to NaN
        rt = mgf_spec['params'].get('rtinseconds', np.nan)

        # try to parse scan number and run_name from title
        title = mgf_spec['params'].get('title', '')
        run_name_match = re.search(self._re_run_name, title)
        try:
            run_name = run_name_match.group(1)
        except AttributeError:
            run_name = self.default_run_name

        scan_number_match = re.search(self._re_scan_number, title)
        try:
            scan_number = int(scan_number_match.group(1))
        except (AttributeError, ValueError):
            scan_number = -1

        return Spectrum(precursor, mgf_spec['m/z array'], mgf_spec['intensity array'], scan_id,
                        rt, self.file_name, self.source_path, run_name, scan_number,
                        scan_index, title=title)

    @property
    def spectra(self):
        """Generator wrapped around pyteomics generator. Reformatting the spectrum information."""
        if (self.step == 1 and self.offset == 0):
            # just forward everything
            for scan_index, mgf_spec in enumerate(self._reader):
                yield self._convert_spectrum(scan_index, mgf_spec)
        else:
            # forward only every nth spectrum
            count = -self.offset
            for scan_index, mgf_spec in enumerate(self._reader):
                if count >= 0 and count % self.step == 0:
                    yield self._convert_spectrum(scan_index, mgf_spec)
                count += 1


class MZMLReader(SpectraReader):
    """SpectraReader for mzML files."""

    def load(self, source, file_name=None, source_path=None, offset=0, step=1):
        """
        Read in spectra from an mzML file and stores them as Spectrum objects.

        :param source: file source, path or stream
        :param file_name: (str) mzML filename
        :param source_path: (str) path to the source file (mzML or archive)
        """
        self.offset = offset
        self.step = step
        self._reader = mzml.read(source)
        if self._reader.index is None:
            self._reader = mzml.read(source, use_index=True)
        super().load(source, file_name, source_path)

        # get the default run name
        if issubclass(type(self._source), tarfile.ExFileObject) or \
                issubclass(type(self._source), zipfile.ZipExtFile):
            text = self._source.read()
            result = re.finditer(b'defaultSourceFileRef="(.*)"', text)
            try:
                result = result.__next__().groups()
            except StopIteration:
                result = None
            self._source.seek(0)
        else:
            with open(self._source, 'r+') as f:
                text = mmap.mmap(f.fileno(), 0)
                result = re.finditer(b'defaultSourceFileRef="(.*)"', text)
                try:
                    result = result.__next__().groups()
                except StopIteration:
                    result = None
                text.close()
        if result is not None:
            self.default_run_name = self._reader.get_by_id(result[0].decode('ascii'))['name']
        else:
            # try to get the default run name from sourceFileList:
            try:
                source_files = list(self._reader.iterfind('//sourceFileList'))[0]
                # if there is more than one entry we can't determine the default
                if source_files['count'] != 1:
                    self.default_run_name = file_name
                else:
                    self.default_run_name = source_files['sourceFile'][0]['name']
            except XMLSyntaxError:
                self.default_run_name = file_name
            self.reset()

    def reset(self):
        """Reset the reader."""
        if issubclass(type(self._source), tarfile.ExFileObject) or \
                issubclass(type(self._source), zipfile.ZipExtFile):
            self._source.seek(0)
            self._reader = mzml.read(self._source)
        else:
            self._reader.reset()

    def count_spectra(self):
        """
        Count the number of spectra.

        :return (int) Number of spectra in the file.
        """
        count_from_offset = len(self._reader) - self.offset
        return int(count_from_offset/self.step) + (count_from_offset % self.step > 0)

    def _convert_spectrum(self, spec):

        # check for single scan per spectrum
        if spec['scanList']['count'] != 1:
            raise ValueError(
                "xiSEARCH2 currently only supports a single scan per spectrum.")
        scan = spec['scanList']['scan'][0]

        # check for single precursor per spectrum
        if spec['precursorList']['count'] != 1 or \
                spec['precursorList']['precursor'][0]['selectedIonList']['count'] != 1:
            raise ValueError(
                "xiSEARCH2 currently only supports a single precursor per spectrum.")
        p = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]

        # create precursor dict
        precursor = {
            'mz': p['selected ion m/z'],
            'charge': p.get('charge state', np.nan),
            'intensity': p.get('peak intensity', np.nan)
        }

        # id is required in mzML so set this as scan_id
        scan_id = spec['id']

        # index is also required in mzML so just use this
        scan_index = spec['index']

        # parse retention time, default to NaN
        rt = scan.get('scan start time', np.nan)
        rt = rt * 60

        # sourceFileRef can optionally reference the 'id' of the appropriate sourceFile.
        if hasattr(spec, 'sourceFileRef'):
            run_name = self._reader.get_element_by_id(spec['sourceFileRef'])['name']
        else:
            run_name = self.default_run_name

        # try to parse scan number from scan_id
        scan_number_match = re.search(self._re_scan_number, scan_id)
        try:
            scan_number = int(scan_number_match.group(1))
        except (AttributeError, ValueError):
            scan_number = None

        return Spectrum(precursor, spec['m/z array'], spec['intensity array'], scan_id,
                        rt, self.file_name, self.source_path, run_name, scan_number, scan_index)

    @property
    def spectra(self):
        """Spectra generator wrapped around pyteomics generator."""

        if (self.step == 1 and self.offset == 0):
            # just forward everything
            for spec in self._reader:
                # skip non-MS2
                if spec['ms level'] != 2:
                    continue
                yield self._convert_spectrum(spec)
        else:
            # forward only every nth spectrum
            count = -self.offset
            for spec in self._reader:
                if spec['ms level'] != 2:
                    continue
                if count >= 0 and count % self.step == 0:
                    yield self._convert_spectrum(spec)
                count += 1
            count = -1
