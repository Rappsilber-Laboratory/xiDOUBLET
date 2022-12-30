import numpy as np
from xidoublet import const, dtypes
from copy import copy
from xidoublet.filters.base_filter import BaseFilter


class IsotopeDetector(BaseFilter):
    """
    Filter to find the isotope clusters in a spectrum.

    Isotope clusters are stored in the `isotope_clusters_mz`, `isotope_clusters_charge` and
    `isotope_clusters_peaks` properties.
    """

    # factors used to estimate the isotopic intensities depending on mass
    AVERAGINE_A = np.array([-0.02576, 0.049889, 0.029321, 0.020406, 0.012126, 0.013333, 0.006667,
                            0.003333, 0.001667])
    AVERAGINE_B = np.array([136, 0, -2.93, -2.04, -12.0, -26.0, -39, -58, -87])

    AVERAGINE_A_LAST = AVERAGINE_A[-1]
    AVERAGINE_B_LAST = AVERAGINE_B[-1]

    AVERAGINE_length = len(AVERAGINE_A)

    # by what mass does the mono-isotopic peak officially "disappear"
    BORDER_MASS = -AVERAGINE_B[0] / AVERAGINE_A[0]

    breakup_factor = 4

    class Cluster:
        """Internal representation of an isotope cluster."""

        def __init__(self, spectrum, charge, peaks, overlap=False):
            """Initialise the values - and determines length of the cluster."""
            self.mz = spectrum.mz_values[peaks[0]]
            self.charge = charge
            self.overlap = overlap
            self.peaks = peaks
            self.length = len(peaks)

    def __init__(self, context):
        """Initialise the IsotopeDetector."""
        BaseFilter.__init__(self, context)
        self.breakup_factor = context.config.isotope_config.avergine_breakup_factor
        self.min_cluster_size = context.config.isotope_config.avergine_min_cluster_size
        self.max_mono_to_first_peak_ratio = \
            context.config.isotope_config.max_mono_to_first_peak_ratio
        self.cluster_intensity_func = context.config.isotope_config.cluster_intensity

    def process(self, spectrum):
        """
        Search for isotope clusters and returns an isotope detected version.

        :param spectrum: (Spectrum) Spectrum to process
        :return: (Spectrum) Isotope detected copy of the spectrum
        """
        # this makes a shallow copy of the spectrum, so any attributes
        # that aren't changed (like the mz_values and int_values) are shared
        new_spec = copy(spectrum)
        new_spec.peak_has_cluster = np.zeros(len(new_spec.mz_values), dtype=bool)

        all_isotope_cluster = []
        accepted_cluster = []
        for charge in range(1, spectrum.precursor_charge+1):
            all_isotope_cluster.extend(self.find_isotope_clusters_for_charge(new_spec, spectrum,
                                                                             charge))

        all_isotope_cluster.sort(key=lambda c: (c.mz, -c.charge))
        # check the cluster
        if len(all_isotope_cluster) > 0:
            accepted_cluster = self.filter_cluster(all_isotope_cluster, new_spec)

        if len(accepted_cluster) > 0:
            # split cluster according to averagine prediction
            deconvoluted = self.deconvolute_isotops(new_spec, accepted_cluster)
            accepted_cluster.extend(deconvoluted)

        # assign each peak that is not part of a cluster a cluster on its own with charge 0
        accepted_cluster.extend([self.Cluster(peaks=np.array([x]), spectrum=spectrum, charge=0)
                                 for x in np.where(~new_spec.peak_has_cluster)[0]])

        accepted_cluster = sorted(accepted_cluster, key=lambda c: (c.mz, c.length, -c.charge))
        # convert into two numpy arrays
        # basic infos

        new_spec.isotope_cluster_charge_values = np.array([c.charge for c in
                                                           accepted_cluster])

        new_spec.isotope_cluster_mz_values = np.array([c.mz for c in
                                                       accepted_cluster])

        if self.cluster_intensity_func == 'max':
            new_spec.isotope_cluster_intensity_values = np.array(
                [np.max(spectrum.int_values[c.peaks]) for c in accepted_cluster])
        elif self.cluster_intensity_func == 'sum':
            new_spec.isotope_cluster_intensity_values = np.array(
                [np.sum(spectrum.int_values[c.peaks]) for c in accepted_cluster])

        new_spec.isotope_cluster_peaks = np.array([(cid, pid)
                                                   for cid in range(len(accepted_cluster))
                                                   for pid in accepted_cluster[cid].peaks],
                                                  dtype=dtypes.peak_cluster)
        new_spec.isotope_cluster = accepted_cluster
        # peak assignment

        return new_spec

    def filter_cluster(self, clusters, new_spec):
        """
        Filter out cluster that are wholly contained in other cluster.

        Cluster that where the second peak does not overlap are accepted - but duplicated to
        cluster that start at the second peak.

        :param clusters: (list(Cluster)) the isotope cluster to test
        :param new_spec: (Spectrum) the spectrum that the cluster should belong to
        """
        accepted_cluster = []
        first_peak = clusters[0].peaks[0]
        for cluster in clusters:
            # if the first peak is not yet registered, or we just start at the same peak as the
            # last cluster (different charge state)
            if ~new_spec.peak_has_cluster[cluster.peaks[0]] or cluster.peaks[0] == first_peak:
                # we accept different charge states starting at the same peak
                accepted_cluster.append(cluster)
                new_spec.peak_has_cluster[cluster.peaks] = True
                first_peak = cluster.peaks[0]
            else:
                # do we have some new peaks not yet part of a cluster?
                non_overlap = np.where(~new_spec.peak_has_cluster[cluster.peaks])[0]
                if len(non_overlap) > 1:
                    # accept that we have overlapping peaks
                    accepted_cluster.append(cluster)
                    new_spec.peak_has_cluster[cluster.peaks] = True
                    # this only checks first peak being overlapping
                    cluster.overlap = True

                    # but also assume that it could be legitimately starting at the first
                    # non-overlapping
                    if len(non_overlap) > 2:
                        skip_cluster = self.Cluster(new_spec, cluster.charge,
                                                    cluster.peaks[non_overlap[0]:])
                        accepted_cluster.append(skip_cluster)

        return accepted_cluster

    def find_isotope_clusters_for_charge(self, output_spectrum, spectrum, charge=1):
        """
        Search for isotope clusters for a given charge.

        The process is as follows:
        1. We look for groups of peaks that match the expected isotope clusters for all peaks. At
         this point the matches potentially include multiple peaks for a given isotope mass.
        2. We select the peaks best matching the expected isotope masses. This is done in the
        `filter_cluster` method.
        3. All other clusters are stored on the object.

        :param output_spectrum: (Spectrum) Output spectrum where isotope cluster info is saved
        :param spectrum: (Spectrum) Input spectrum
        :param charge: (int) charge state of the clusters to search for
        """
        masses = spectrum.mz_values

        # build an array of all potential masses upto the configured
        # m/z over base, where each of the elements of the array is an
        # column vector. We need column vectors to do the matching
        # with the masses below.
        expected_isotope_masses = self.isotope_masses(
            masses.reshape(1, -1), charge)[:, :, np.newaxis]
        # this matches each of the isotope mass column vectors with the
        # masses vector, producing an array of matrices, where the
        # matrix at index N corresponds to candidates for the Nth
        # isotope for each peak in the spectrum. So if
        # all_matches[3,0,5] is true, that means that the 6th peak in
        # the spectrum matches the mass of the the +3/q isotope of the
        # first peak in the spectrum
        all_matches = np.isclose(masses, expected_isotope_masses,
                                 rtol=2 * self.config.isotope_config.rtol, atol=0)

        # overwrite first 2d matrix with diagonal True entries to make sure each isotope cluster
        # has a single starting peak
        all_matches[0] = False
        np.fill_diagonal(all_matches[0], True)

        # convert the boolean values to indices on the mass array, throwing blank lines away
        raw_clusters = [np.where(cluster)[0] for cluster in np.any(all_matches, axis=0)]

        # raw clusters are not necessarily continuously matched!
        # This gives the position of the first occurrence of a missing match for each
        # cluster i.e. the continuous cluster length
        cluster_len = np.all(~all_matches.T, axis=0).argmax(axis=1)
        # replace 0s with cluster_calc_size
        cluster_len[cluster_len == 0] = self.config.isotope_config.cluster_calc_size

        # Clusters potentially contain multiple peaks per expected isotope mass
        # In the following we filter those out to only leave the best matching peaks.
        # Additionally, we truncate the clusters to continuous clusters ([:l])
        # and throw out single peaks (l > 1)
        all_clusters = [self.filter_cluster_peaks(cluster, masses, charge)[:l]
                        for cluster, l in zip(raw_clusters, cluster_len)
                        if l > 1]

        # Next we check all clusters for artifacts and overlapping clusters
        # (a peak can only belong to a single cluster of the same charge state)

        # create an array saving which cluster a peak belongs to (-1 for no cluster)
        peak_to_cluster = np.full((len(output_spectrum.mz_values)), -1)

        accepted_cluster = []
        for cluster in all_clusters:
            # id of the monoisotopic peak (MIP)
            mip_id = cluster[0]
            # id of the first isotopic peak (FIP)
            fip_id = cluster[1]
            # if the MIP is not already part of a cluster check the cluster for artifacts
            if peak_to_cluster[mip_id] == -1:
                # We do this by comparing the intensity ratio of the assumed monoisotopic peak
                # to the first isotopic peak. If the MIP is much smaller then next peak we are
                # probably looking at an artifact and don't accept the cluster
                if spectrum.int_values[fip_id] > spectrum.int_values[mip_id] *\
                        self.max_mono_to_first_peak_ratio:
                    continue
                return_cluster = self.Cluster(peaks=cluster, spectrum=spectrum,
                                              charge=charge)
                # mark all peaks as belonging to a cluster
                peak_to_cluster[cluster] = len(accepted_cluster)
                accepted_cluster.append(return_cluster)

            # if the MIP already belongs to a cluster but the FIP doesn't we are probably looking
            # at overlapping clusters. So we expand the former cluster by the peaks of the current
            # cluster
            elif peak_to_cluster[fip_id] == -1:
                former_cluster_id = peak_to_cluster[mip_id]
                former_cluster = accepted_cluster[former_cluster_id]
                # combine former cluster peaks with new cluster peaks
                former_cluster.peaks = np.hstack([former_cluster.peaks, cluster[1:]])
                former_cluster.length = len(former_cluster.peaks)
                peak_to_cluster[cluster] = peak_to_cluster[mip_id]

        return accepted_cluster

    def filter_cluster_peaks(self, cluster, masses, charge):
        """
        Select the peaks closest to the expected isotope peaks from the cluster.

        :param cluster (ndarray) peak index into masses for the peaks
        :param masses (ndarray) the m/z values for all peaks in the spectrum
        :param charge (int) the charge state of the cluster
        """
        sorted_cluster = masses[cluster]
        base = sorted_cluster[0]
        expected_isotopes = self.isotope_masses(base, charge)
        matches = np.isclose(sorted_cluster, expected_isotopes,
                             rtol=2 * self.config.isotope_config.rtol, atol=0)
        errors = np.abs((sorted_cluster - expected_isotopes) / expected_isotopes)
        best_match_indices = np.argmin(errors, axis=1)
        replacement_indices = np.where(np.sum(matches, axis=1) > 1)[0]
        matches[replacement_indices] = False
        for i in replacement_indices:
            matches[i][best_match_indices[i]] = True
        return cluster[np.where(matches)[1]]

    def isotope_masses(self, mass, charge):
        """
        Calculate the expected isotopes for a given mass.

        :param mass: (float64, ndarray) Any value that numpy can broadcast over, in this case
        typically a single value or an array of masses. When it has an array, it should have a shape
         like (n, 1) and will then return a matrix of shape (n, self.config.max_cluster_size)
        :param charge: (int) The charge for which to calculate the m/z values
        :return: For a single mass, it will return an array of masses for the isotopes.
         For a mass column vector, it will return an array of masses, where each column corresponds
         to the expected isotopes for a mass in the original array.
        """
        increments = np.arange(
            self.config.isotope_config.cluster_calc_size) * const.c12c13_massdiff / charge
        return increments.reshape(-1, 1) + mass

    def relative_height(self, mass):
        """
        Calculate the expected relative height for an isotope cluster of the given mass.

        :param mass (float64) Mass to calculate isotope cluster relative intensities for
        :return isotope cluster intensities for the mass based on averagine
        :rtype (float64, ndarray)
        """
        # TODO turn this into a simple lookup by precomputing from 100 to 10000 in some steps
        #  and only calculate if we exceed that
        ret = mass * self.AVERAGINE_A + self.AVERAGINE_B
        zero = np.where(ret <= 0)
        if len(zero) > 0:
            ret[zero[0]] = ret[zero[0][0]-1]/2

        return ret

    def deconvolute_isotops(self, spectrum, clusters):
        """
        Detect and split up overlapping Isotope-cluster of same charge state.

        :param spectrum: the spectrum that contains the clusters
        :param clusters: all clusters that should be checked
        :return: a list of newly generated isotope-clusters
        """
        new_cluster = []
        for cluster in clusters:
            new_cluster.extend(self.deconvolute_single_cluster(spectrum, cluster,
                                                               self.breakup_factor))
        return new_cluster

    def deconvolute_single_cluster(self, spectrum, cluster, breakup_limit):
        """
        Test a single isotope cluster, if we can detect an overlapping cluster.

        :param spectrum: the spectrum that contains the cluster
        :param cluster: cluster that should be checked
        :param breakup_limit: a deviation from the predicted distribution larger then this
        factor will initiate a break
        :return: list of newly generated clusters
        """
        clength = cluster.length
        # small clusters are not checked
        # and also if we previously extended this cluster to lower masses
        # this check will not make sense
        if clength < self.min_cluster_size or cluster.overlap:
            return []

        # approximate mass
        mass = spectrum.mz_values[cluster.peaks[0]] * cluster.charge
        intensities = spectrum.int_values[cluster.peaks]

        # get the expected distribution
        expect_intens = self.relative_height(mass)
        # scale to first observed peak
        # TODO: change to most abundant peak as reference
        expect_intens = expect_intens * intensities[0] / expect_intens[0]

        # if it is to long we just duplicate the last value
        # TODO better would be to make a gradient down to zero
        if (clength > self.AVERAGINE_length):
            expect_intens = np.concatenate(
                (expect_intens, np.repeat(expect_intens[-1]/2, clength - self.AVERAGINE_length)))
        elif clength < self.AVERAGINE_length:
            expect_intens = expect_intens[:clength]

        ratio = np.maximum(intensities/expect_intens, expect_intens/intensities)

        breakup = np.where(ratio > breakup_limit)[0]

        # if we exceed the limits we will create a new cluster starting from there
        if len(breakup) > 0:
            new_peaks = cluster.peaks[breakup[0]:]
            copy_cluster = self.Cluster(peaks=new_peaks, spectrum=spectrum,
                                        charge=cluster.charge)
            new_cluster = [copy_cluster]

            # also test the remaining cluster, to see if it should be broken up as well
            # NOTE: this different to xi1
            new_cluster.extend(self.deconvolute_single_cluster(spectrum, copy_cluster,
                                                               breakup_limit * 1.5))

            return new_cluster
        return []
