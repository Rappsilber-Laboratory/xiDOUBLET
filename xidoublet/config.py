"""Configuration file for running xisearch."""
import json
import re
import copy

# Unique sentinel, used to allow None to be a valid default for a setting.
NO_DEFAULT = object()


def stringhash(s):
    """
    Create a stable hash code for strings.

    When run in debugger hash(str) seems not give a stable result -
    as after restarting the debugger hash(str) with exactly the same string
    gives a different value.
    """
    return hash(tuple(ord(x) for x in s))


class Setting:
    """A setting supported by the config system."""

    def __init__(self, type, default=NO_DEFAULT, valid_values=None, required=True, max_value=None):
        """
        Initialise the Setting.

        :param type: Python type expected for this setting.
        :param default: Default value for this setting.
        :param valid_values: Tuple of accepted values, re pattern, or None to accept any value.
        :param max_value: Maximum value of setting (for float or int types)
        """
        self.type = type
        self.valid_values = valid_values
        if max_value is not None and not any([issubclass(self.type, int),
                                              issubclass(self.type, float)]):
            raise TypeError("max_value is only supported for int and float type.")
        self.max_value = max_value
        if default is not NO_DEFAULT:
            try:
                self.default = self.accept(default)
            except TypeError:
                raise TypeError("Default '%s' is not valid and could not be coerced "
                                "into the expected type (%s)" % (repr(default),
                                                                 repr(self.type))) from None
            self.required = False
        else:
            self.required = required

    def accept(self, value):
        """
        Check if a value has a value is valid.

        :param value: (mixed) value to check
        :return: (bool) True if valid, else False
        """
        coerced_value = self.coerce(value)
        if self.max_value is not None and coerced_value >= self.max_value:
            raise ValueError(f'{coerced_value} is above max_value({self.max_value})!')
        if self.valid_values is not None:
            if isinstance(self.valid_values, re.Pattern):
                if self.valid_values.match(coerced_value) is None:
                    raise ValueError(f'{coerced_value} is not valid!'
                                     f' Valid values need to match: {self.valid_values.pattern}')
            elif coerced_value not in self.valid_values:
                raise ValueError(f'{coerced_value} is not valid!'
                                 f' Valid values are: {self.valid_values}')
        return coerced_value

    def coerce(self, value):
        """
        Coerce a value into the correct type.

        :param value: (mixed) value to coerce
        :return: (mixed) coerced value
        """
        if isinstance(value, self.type):
            return value
        try:
            if isinstance(value, dict):
                return self.type(**value)
            elif value in self.type.__dict__:
                return self.type.__dict__[value]
            else:
                return self.type(value)
        except ValueError:
            raise TypeError from None

    def hash(self, value):
        """
        Create a hash for a value.

        :param value: (mixed) value to create hash for
        :return: (str) hash of the value
        """
        if issubclass(self.type, ConfigGroup):
            return value.hash()
        elif issubclass(self.type, str):
            return stringhash(value)
        else:
            return hash(value)


class ListSetting(Setting):
    """A Setting with a list of values supported by the config system."""

    def accept(self, values):
        """
        Check if all elements of the ListSetting have valid values.

        :param values: (list) values to check
        :return: (bool) True if valid, else False
        """
        return_list = []

        if not isinstance(values, list):
            if isinstance(self.type, Setting):
                value = self.type.accept(values)
                return_list.append(value)
            else:
                value = super().accept(values)
                return_list.append(value)
        else:
            # if type is a ListSetting again - we make some assumptions
            if isinstance(self.type, ListSetting):
                # if we have a list of lists all is fine
                if any([isinstance(v, list) for v in values]):
                    for value in values:
                        value = self.type.accept(value)
                        return_list.append(value)
                else:
                    # we have just a single list - assume
                    # the list is actually the first entry in the outer list
                    value = self.type.accept(values)
                    return_list.append(value)

            # if the "type" is actually a Setting then use that to convert the value
            elif isinstance(self.type, Setting):
                for value in values:
                    value = self.type.accept(value)
                    return_list.append(value)
            else:
                for value in values:
                    value = super().accept(value)
                    return_list.append(value)

        return return_list

    def hash(self, value):
        """
        Create a hash.

        :return: (int) hash
        """
        if isinstance(self.type, Setting):
            return hash(tuple(self.type.hash(x) for x in value))
        elif issubclass(self.type, ConfigGroup):
            return hash(tuple(x.hash() for x in value))
        elif issubclass(self.type, str):
            return hash(tuple(stringhash(x) for x in value))
        else:
            return hash(tuple(value))


class ConfigMeta(type):
    """Metaclass used to define configuration groups."""

    def __new__(cls, name, bases, attributes):
        """Create a new instance."""
        settings = {k: a for k, a in attributes.items() if isinstance(a, Setting)}
        others = {k: a for k, a in attributes.items() if k not in settings}
        defaults = {k: s.default for k, s in settings.items() if hasattr(s, 'default')}
        required = set([k for k, s in settings.items() if s.required])
        new_attributes = dict(_settings=attributes, _defaults=defaults, _required=required,
                              **others)
        return type.__new__(cls, name, bases, new_attributes)


class ConfigGroup(metaclass=ConfigMeta):
    """Base class for configuration groups."""

    def __init__(self, **kwargs):
        """Initialise the ConfigGroup."""
        self._values = {}
        for key, value in kwargs.items():
            if key not in self._settings:
                raise KeyError("Unknown setting '%s'" % key)
            setattr(self, key, value)

        # transfer defaults to those values that are not set explicitly
        for k, v in self._defaults.items():
            if k not in kwargs.keys():
                setattr(self, k, copy.deepcopy(self._defaults[k]))

        for setting in self._required:
            if setting not in kwargs.keys():
                raise AttributeError("'%s' is required but not defined" % setting) from None

    def __setattr__(self, key, value):
        """Set the value of a Setting."""
        if key.startswith('_') or key not in self._settings:
            super(ConfigGroup, self).__setattr__(key, value)
            return
        setting = self._settings[key]
        try:
            self._values[key] = setting.accept(value)
        except TypeError:
            raise TypeError("Value '%s' is not valid for '%s' and could not be coerced "
                            "into the expected type (%s)" % (repr(value), key,
                                                             repr(setting.type))) from None
        except ValueError:
            raise ValueError("Value '%s' is not valid for '%s'" % (repr(value), key)) from None

    def __contains__(self, key):
        """Check if a Setting is configured in the ConfigGroup."""
        return key in self._settings

    def __getattr__(self, key):
        """Get the value for a Setting."""
        if key.startswith('_') or key not in self._settings:
            return super(ConfigGroup, self).__getattr__(key)
        elif key in self._values:
            return self._values[key]
        else:
            raise AttributeError(key)

    def hash(self):
        """
        Create a hash.

        :return: (int) hash
        """
        value_hashes = [(stringhash(name), self._settings[name].hash(value))
                        for name, value in self._values.items()]
        return hash(frozenset(value_hashes))

    @classmethod
    def from_json(cls, json_string):
        """Create a ConfigGroup from a JSON string."""
        args = json.loads(json_string)
        return cls(**args)


class Stub(ConfigGroup):
    """Stub configuration."""

    """Name (only single lowercase char allowed)"""
    name = Setting(str, valid_values=re.compile('[a-z]$'))

    """Mass in Dalton"""
    mass = Setting(float)

    """Other stub(s) this is connected to"""
    pairs_with = ListSetting(str, default=[])


class Crosslinker(ConfigGroup):
    """Crosslinker configuration."""

    """Name"""
    name = Setting(str, valid_values=re.compile('^.{1,39}$'))

    """Mass in Dalton"""
    mass = Setting(float)

    """Crosslinker stub mod masses (modifications associated with crosslinker cleavage)"""
    cleavage_stubs = ListSetting(Stub, [], required=False)


Crosslinker.SDA = Crosslinker(name='SDA',
                              mass=82.04186484,
                              cleavage_stubs=[
                                  Stub(name='o', mass=0, pairs_with=['s']),
                                  Stub(name='s', mass=82.04186484, pairs_with=['o']),
                              ])

Crosslinker.DSSO = Crosslinker(name='DSSO',
                               mass=158.0038,
                               cleavage_stubs=[
                                   Stub(name='a', mass=54.010565, pairs_with=['s', 't']),
                                   Stub(name='s', mass=103.993200, pairs_with=['a']),
                                   Stub(name='t', mass=85.982635, pairs_with=['a']),
                               ])


class IsotopeDetectorConfig(ConfigGroup):
    """Isotope configuration."""

    """
    Relative tolerance for matching isotope clusters (defaults to ms2_rtol).
    This rtol is assumed for pairwise peak comparison on both peaks. Currently approximated by
    2 * rtol on the first peak.
    """
    rtol = Setting(float, -1)

    """
    Maximum Number of isotope peaks to look for in the initial state.
    Longer cluster will still be recognised - but in two steps.
    """
    cluster_calc_size = Setting(int, 7)

    """Only assume the start of a cluster if the ratio of the first peak to the second does not
    exceed this ratio. Stems mainly from observations with labeled peptides/crosslinker."""
    max_mono_to_first_peak_ratio = 8

    """Only try to break up clusters that are longer than this"""
    avergine_min_cluster_size = Setting(int, 5)

    """When the intensity of a peak is off by this factor assume a new cluster starts"""
    avergine_breakup_factor = Setting(int, 5)

    """How to calculate isotope cluster intensity"""
    cluster_intensity = Setting(str, 'max', valid_values=('max', 'sum'))  # ToDo: sum default?


class DoubletConfig(ConfigGroup):
    """Cleavable crosslinker doublet detection related config."""

    def __init__(self, **kwargs):
        """Initialise and check validity of the Config."""
        super().__init__(**kwargs)

        if len(self.stubs) != 2:
            raise ValueError("You need to define exactly 2 stubs to use.")

    """Doublet rank cutoff. Higher intense peak of the doublet determines the doublet rank."""
    rank_cutoff = Setting(int, -1)

    """Crosslinker stubs (by name) to use for detection. (atm only 2 stubs are supported)."""
    stubs = ListSetting(str, ['a', 't'])

    """Match also peaks with undefined charge states"""
    undefined_charge_match = Setting(bool, False)

    """Cap the number of doublets to be detected"""
    cap = Setting(int, -1)

    """Filter out doublets by remaining mass of the second peptide."""
    second_peptide_mass_filter = Setting(float, -1)


class Config(ConfigGroup):
    """Top level configuration for a search."""

    def __init__(self, **kwargs):
        """
        Initialise the Config.

        Forwards all kwargs to super().__init__ and translates loss specificity and crosslinker
        specificities. Also does some validity checks.
        """
        super().__init__(**kwargs)

        # translate textual ms tolerances to numeric atol, rtol values
        re_ms_tol = re.compile(r"([0-9.]+)\s*(da|th|ppm)", re.IGNORECASE)

        def translate_ms_tol(str_tol):
            """Translate a tolerance string into atol/rtol."""
            atol, rtol = 0, 0
            tol, unit = re_ms_tol.search(str_tol).groups()

            if unit.lower() == 'da' or unit.lower() == 'th':
                atol = float(tol)
            elif unit.lower() == 'ppm':
                rtol = float(tol) * 1e-6

            return atol, rtol

        self.ms2_atol, self.ms2_rtol = translate_ms_tol(self.ms2_tol)

        if self.ms2_atol > 0.06 or self.ms2_rtol > 5e-5:
            self.low_resolution = True
        else:
            self.low_resolution = False

        # if isotope tolerance is not explicitly set use the ms2 relative tolerance
        if self.isotope_config.rtol < 0:
            self.isotope_config.rtol = self.ms2_rtol

    """
    Max number of threads to use for processing spectra. Setting to 0 means using the
    multiprocessing default, which is the value of `cpu_count`. Setting it to a negative
    number N means use all but minus N threads.
    """
    threads = Setting(int, 0)

    _re_ms_tol = re.compile(r'^[0-9]+(?:.[0-9]+)?\s*(?:ppm|th|da)$', re.IGNORECASE)

    """Tolerance for matching fragment (MS2) m/z values."""
    ms2_tol = Setting(str, '10 ppm', valid_values=_re_ms_tol)

    """Crosslinker in use"""
    crosslinker = Setting(Crosslinker, Crosslinker.DSSO)

    """Isotope processor config"""
    isotope_config = Setting(IsotopeDetectorConfig, IsotopeDetectorConfig())

    """Regular expression used for matching the scan number"""
    re_scan_number = Setting(str, "(?:scan=|[^.]*\\.)([0-9]+)(?:\\.\1)?")

    """Regular expression used for matching the run name"""
    re_run_name = Setting(str, "^([^\\s.]+)")

    """doublet matchting related configs"""
    doublet = Setting(DoubletConfig, DoubletConfig())


class ConfigReader:
    """Config Reader class."""

    @classmethod
    def load_file(cls, file_name):
        """Open a file by filename and create a Config from it."""
        with open(file_name) as f:
            return cls.load(f)

    @classmethod
    def load(cls, fp):
        """Create a Config from a JSON string."""
        settings = json.load(fp)
        config = Config(**settings)
        return config
