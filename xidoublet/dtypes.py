"""Central place for reused numpy data types."""
import numpy as np

peak_cluster = np.dtype([
    ('cluster_id', np.uint16),      # id of the cluster
    ('peak_id', np.uint16)          # a peak that belongs to the cluster
])

doublet_dtype = np.dtype([
    ('peak0_mz', np.float64),
    ('peak1_mz', np.float64),
    ('peak0_rel_int', np.float32),
    ('peak1_rel_int', np.float32),
    ('peak0_rank', np.int_),
    ('peak1_rank', np.int_),
    ('charge', np.uint8),
    ('peptide_mass', np.float64),
    ('2nd_peptide_mass', np.float64),
    ('abs_error', np.float64),
    ('rel_error', np.float64),
])
