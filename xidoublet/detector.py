from xidoublet.filters import IsotopeDetector, IsotopeReducer, RelativeIntensityFilter, \
    IntensityRankFilter
from xidoublet import doublet
import numpy as np
from numpy.lib import recfunctions


class SearchContext:
    """Context that holds related data for processing spectra (config and filters)."""
    def __init__(self, config):
        """
        Initialise the search context class.

        :param config: (Configuration) search configuration
        """
        self.config = config
        self.crosslinker = config.crosslinker
        # init spectrum filters
        self.isotope_detector = IsotopeDetector(self)
        self.isotope_reducer = IsotopeReducer()

        if config.doublet.intensity_cutoff != -1:
            self.cutoff = config.doublet.intensity_cutoff
            self.noise_filter = RelativeIntensityFilter()
        elif config.doublet.rank_cutoff != -1:
            self.cutoff = config.doublet.rank_cutoff
            self.noise_filter = IntensityRankFilter()
        else:
            self.cutoff = None
            self.noise_filter = None

        # setup stub masses
        stub1 = next((x for x in self.crosslinker.cleavage_stubs
                      if x.name == config.doublet.stubs[0]), None)
        stub2 = next((x for x in self.crosslinker.cleavage_stubs
                      if x.name == config.doublet.stubs[1]), None)
        stubs = [stub1, stub2]
        if any(stubs) is None:
            raise ValueError(
                "Configured stub names: {} not found in crosslinker stubs ({}).".format(
                    ', '.join(config.doublet.stubs),
                    ', '.join([s.name for s in self.crosslinker.cleavage_stubs])
                )
            )

        stubs.sort(key=lambda s: s.mass)
        self.stub_combinations = {
            '_'.join([s.name for s in stubs]): (stubs[0].mass, stubs[1].mass),
        }


class SpectrumProcessor:
    """Class for processing spectra."""
    def __init__(self, context):
        """
        Initialise the spectrum processor class.

        :param context: (SearchContext) search context
        """
        self.context = context

    def process_spectrum(self, spectrum, doublet_queue):
        """
        Process a single spectrum and push the results to the result queue.

        :param spectrum: (Spectrum) a MS2 spectrum object
        :param doublet_queue: (Queue) queue to which the doublet results can be pushed
        """
        context = self.context
        config = context.config.doublet
        base_table_dtype = [
            ('scan_index', np.uint),
            ('scan_number', np.uint),
            ('file_name', np.dtype('<U%d' % len(spectrum.file_name))),
            ('scan_id', np.dtype(('<U%d' % len(spectrum.scan_id))))
        ]

        # spectrum processing
        # rel intensity or rank cutoff before isotope detection
        if config.cutoff_point == 'beforeDeisotoping' and context.noise_filter is not None:
            filtered_spec = context.noise_filter.process(spectrum, context.cutoff)
        else:
            filtered_spec = spectrum

        # Isotope detection and reduction
        if context.config.low_resolution:
            # filtered spectrum
            doublet_spec = filtered_spec
            # set isotope cluster related values (they are used for downstream processing)
            doublet_spec.charge_values = np.zeros(doublet_spec.mz_values.size)
            doublet_spec.isotope_cluster_mz_values = doublet_spec.mz_values
            doublet_spec.isotope_cluster_charge_values = doublet_spec.charge_values
            doublet_spec.isotope_cluster_intensity_values = doublet_spec.int_values

        else:
            # filtered spectrum
            isotope_detected = context.isotope_detector.process(filtered_spec)
            isotope_reduced = context.isotope_reducer.process(isotope_detected)
            doublet_spec = isotope_reduced

        # rel intensity or rank cutoff after isotope detection
        if config.cutoff_point == 'afterDeisotoping' and context.noise_filter is not None:
            doublet_spec = context.noise_filter.process(doublet_spec, context.cutoff)

        doublets = []
        for stub_comb_name, (small_stub, large_stub) in context.stub_combinations.items():
            d = doublet.find_doublets(doublet_spec, small_stub, large_stub, context)
            if d.size > 0:
                # generate base table
                table = np.empty(
                    d.size, dtype=base_table_dtype + [('stub_combination', np.dtype('<U7'))])
                table['scan_index'] = spectrum.scan_index
                table['scan_number'] = spectrum.scan_number
                table['file_name'] = spectrum.file_name
                table['scan_id'] = spectrum.scan_id
                table['stub_combination'] = stub_comb_name
                # write result
                doublets.append(recfunctions.merge_arrays([d, table], flatten=True))

        if len(doublets) > 0:
            doublets_result = np.hstack(doublets)

            # filter by 2nd peptide mass
            if config.second_peptide_mass_filter != -1:
                doublets_result = doublets_result[
                    doublets_result['2nd_peptide_mass'] >= config.second_peptide_mass_filter]

            # cap doublets
            if config.cap != -1:
                doublet_sort = np.argsort(np.fmin(doublets_result['peak0_rank'],
                                                  doublets_result['peak1_rank']))

                doublets_result = doublets_result[doublet_sort][:config.cap]

            # write results
            doublet_queue.put(doublets_result)
