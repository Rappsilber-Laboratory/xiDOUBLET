from xidoublet.filters import IsotopeDetector, IsotopeReducer
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

        # For UCCL/UVPD experiments, we add stub combinations to account for hydrogen shifts
        # deltaM is B_C and B-1_C-1 (almost no B+1_C+1)
        # deltaM+1 is mostly B-1_C and to a lesser extent B_C+1
        # deltaM+2 is actual B-1_C+1
        # deltaM-1 and deltaM-2 are rarely observed
        if not config.doublet.hydrogen_shifts == 'none':
            self.stub_combinations.update({
                # (B_C)+1: could be B-1_C or B_C+1 (use B-1_C here)
                f'({stubs[0].name}_{stubs[1].name})+1':
                    (stubs[0].mass - const.H_MASS, stubs[1].mass),
                # (B_C)+2: could be B-1_C+1 (or B-2_C)
                f'({stubs[0].name}_{stubs[1].name})+2':
                    (stubs[0].mass - const.H_MASS, stubs[1].mass + const.H_MASS),
            })
            if config.doublet.hydrogen_shifts == 'wide':
                self.stub_combinations.update({
                    # (B_C)-1: could be B_C-1 or B+1_C (use B_C-1 here)
                    f'({stubs[0].name}_{stubs[1].name})-1':
                        (stubs[0].mass, stubs[1].mass - const.H_MASS),
                    # (B_C)+3: could be B-1_C+2 or B-2_C+1 (use B-2_C+1 here)
                    f'({stubs[0].name}_{stubs[1].name})+3':
                        (stubs[0].mass - 2*const.H_MASS, stubs[1].mass + const.H_MASS),
                })


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
        # Isotope detection and reduction
        if context.config.low_resolution:
            # filtered spectrum
            doublet_spec = spectrum
            # set isotope cluster related values (they are used for downstream processing)
            doublet_spec.charge_values = np.zeros(doublet_spec.mz_values.size)
            doublet_spec.isotope_cluster_mz_values = doublet_spec.mz_values
            doublet_spec.isotope_cluster_charge_values = doublet_spec.charge_values
            doublet_spec.isotope_cluster_intensity_values = doublet_spec.int_values

        else:
            # filtered spectrum
            isotope_detected = context.isotope_detector.process(spectrum)
            isotope_reduced = context.isotope_reducer.process(isotope_detected)
            doublet_spec = isotope_reduced

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

            # filter by doublet rank
            if config.rank_cutoff != -1:
                doublet_rank = np.fmin(doublets_result['peak0_rank'], doublets_result['peak1_rank'])
                doublets_result = doublets_result[doublet_rank <= config.rank_cutoff]

            # filter by 2nd peptide mass
            if config.second_peptide_mass_filter != -1:
                doublets_result = doublets_result[
                    doublets_result['2nd_peptide_mass'] >= config.second_peptide_mass_filter]

            # filter to single doublet per m/z window
            if config.mz_window_filter != -1:
                doublet_sort = np.argsort(np.fmin(doublets_result['peak0_rank'],
                                                  doublets_result['peak1_rank']))

                peak0_mask = doublets_result['peak0_rank'] < doublets_result['peak1_rank']
                trigger_mz = np.empty(doublets_result.size, dtype=np.float64)
                trigger_mz[peak0_mask] = doublets_result['peak0_mz'][peak0_mask]
                trigger_mz[~peak0_mask] = doublets_result['peak1_mz'][~peak0_mask]
                doublet_idx = [doublet_sort[0]]
                for i in range(1, doublet_sort.size):
                    if not any(np.isclose(trigger_mz[doublet_sort[i]], trigger_mz[doublet_idx],
                                          atol=config.mz_window_filter)):
                        doublet_idx.append(doublet_sort[i])
                doublets_result = doublets_result[doublet_idx]

            # cap doublets
            if config.cap != -1:
                doublet_sort = np.argsort(np.fmin(doublets_result['peak0_rank'],
                                                  doublets_result['peak1_rank']))

                doublets_result = doublets_result[doublet_sort][:config.cap]

            # write results
            doublet_queue.put(doublets_result)
