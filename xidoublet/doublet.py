import numpy as np
from xidoublet import const, dtypes


def find_doublets(spectrum, s_stub_mass, l_stub_mass, context):
    """
    Find doublets of peaks in a spectrum.

    :param spectrum: (Spectrum) spectrum to search
    :param s_stub_mass: (float) stub mass of the smaller stub
    :param l_stub_mass: (float) stub mass of the larger stub
    :param context: (Searcher) Search context
    :return: array of matched doublets
    :rtype: numpy.ndarray
        peak0_mz: (np.float64) m/z value of peak 0
        peak1_mz: (np.float64) m/z value of peak 1
        peak0_rel_int: (np.float32) intensity of peak 0 relative to base peak of this
            (processed) spectrum
        peak1_rel_int: (np.float32) intensity of peak 0 relative to base peak of this
            (processed) spectrum
        charge: (np.uint8) charge state of the doublet
        peptide_mass (np.float64) mass of the peptide
        abs_error: (np.float64) absolute error of the match
        rel_error: (np.float64) relative error of the match
    """
    config = context.config

    delta_mass = l_stub_mass - s_stub_mass

    defined_charge_mask = spectrum.charge_values != 0

    if config.doublet.undefined_charge_match:
        # repeat undefined charge state peaks for each charge state up to precursor charge - 1
        undef_mz_values = np.repeat(spectrum.mz_values[~defined_charge_mask],
                                    spectrum.precursor_charge-1)
        undef_int_values = np.repeat(spectrum.int_values[~defined_charge_mask],
                                     spectrum.precursor_charge-1)
        undef_charge_values = np.tile(np.arange(1, spectrum.precursor_charge),
                                      spectrum.mz_values[~defined_charge_mask].size)

        mz_values = np.hstack([spectrum.mz_values[defined_charge_mask], undef_mz_values])
        charge_values = np.hstack([spectrum.charge_values[defined_charge_mask],
                                   undef_charge_values])
        int_values = np.hstack([spectrum.int_values[defined_charge_mask], undef_int_values])

    else:
        mz_values = spectrum.mz_values[defined_charge_mask]
        charge_values = spectrum.charge_values[defined_charge_mask]
        int_values = spectrum.int_values[defined_charge_mask]

    delta_m_mz_values = mz_values + delta_mass / charge_values

    charge_matches = np.equal(charge_values.reshape(-1, 1), charge_values)

    mz_matches = np.isclose(mz_values.reshape(-1, 1), delta_m_mz_values,
                            rtol=config.ms2_rtol, atol=config.ms2_atol)

    # throw out charges larger or equal precursor charge
    precursor_charge_matches = charge_values < spectrum.precursor_charge

    matches = mz_matches & charge_matches & precursor_charge_matches

    matched_mz_indices, matched_delta_mz_indices = np.nonzero(matches)

    upper_peaks_mz = mz_values[matched_mz_indices]
    upper_peaks_int = int_values[matched_mz_indices]
    calc_upper_peaks_mz = delta_m_mz_values[matched_delta_mz_indices]
    # the matched_delta_mz_indices are also the indices into the original spectrum for the
    # lower mz doublet partner
    lower_peaks_mz = mz_values[matched_delta_mz_indices]
    lower_peaks_int = int_values[matched_delta_mz_indices]
    doublet_charges = charge_values[matched_delta_mz_indices]

    # peak ranks
    sorted_intensities = np.sort(spectrum.int_values)
    _, _, int_sorted_ranks = np.unique(sorted_intensities, return_index=True, return_inverse=True)
    # ranks are the wrong way round - change this and convert to 1-based
    int_sorted_ranks = int_sorted_ranks.max(initial=0) - int_sorted_ranks + 1
    # get the indices into the int_sorted_rank array by matching the peak intensities to the sorted
    # intensities. Use argmax because we only want the first match for each intensity
    # (same intensity = same rank)
    upper_peak_rank_idx = np.equal(upper_peaks_int.reshape(-1, 1), sorted_intensities).argmax(
        axis=1)
    lower_peak_rank_idx = np.equal(lower_peaks_int.reshape(-1, 1), sorted_intensities).argmax(
        axis=1)
    upper_peak_ranks = int_sorted_ranks[upper_peak_rank_idx]
    lower_peak_ranks = int_sorted_ranks[lower_peak_rank_idx]

    # calculate the peptide masses of the doublets
    # ToDo: use lower or upper? lower should have smaller mass error
    pep_masses = (lower_peaks_mz - const.proton_mass) * doublet_charges - s_stub_mass
    sec_pep_masses = spectrum.precursor_mass - (pep_masses + context.crosslinker.mass)
    doublet_table = np.empty(matched_mz_indices.size, dtypes.doublet_dtype)

    doublet_table['peak0_mz'] = lower_peaks_mz
    doublet_table['peak1_mz'] = upper_peaks_mz
    base_peak_int = spectrum.int_values.max()
    doublet_table['peak0_rel_int'] = lower_peaks_int / base_peak_int
    doublet_table['peak1_rel_int'] = upper_peaks_int / base_peak_int
    doublet_table['peak0_rank'] = lower_peak_ranks
    doublet_table['peak1_rank'] = upper_peak_ranks
    doublet_table['charge'] = doublet_charges
    doublet_table['peptide_mass'] = pep_masses
    doublet_table['2nd_peptide_mass'] = sec_pep_masses
    doublet_table['abs_error'] = upper_peaks_mz - calc_upper_peaks_mz
    doublet_table['rel_error'] = doublet_table['abs_error'] / calc_upper_peaks_mz

    doublet_table.sort(order='peak0_mz')

    return doublet_table
