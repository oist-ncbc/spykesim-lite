import numpy as np
from clusterer_utils import run_ap_elig


"""
# Notes:

-   Expects the similarity matrix to be 
    symmetric within tolerance (1e-6 since float32)
-   Floor value now fixed to -1. Previously: 
        `floor = np.percentile(self.simmat[mask], 1)`
-   Although this class uses the name 'simmat' for
    its similarity matrix, the user may pass
    a norm. simmat (and is expected to do so
    when passing a meaningful eligibility matrix)
"""


class Clusterer:
    def __init__(self, simmat, eligmat):
        self.simmat = simmat
        self.eligmat = eligmat.astype(bool)

        if self.simmat.shape != self.eligmat.shape:
            raise ValueError("simmat and eligmat must have the same shape")
        
        if self.simmat.ndim != 2 or self.simmat.shape[0] != self.simmat.shape[1]:
            raise ValueError("simmat must be square")
        
        if not np.isfinite(self.simmat).all():
            raise ValueError(f"simmat contains non-finite values")
        
        if not np.allclose(simmat, simmat.T, atol=1e-6):
            raise ValueError(f"simmat is not symmetric within tolerance")

        self.floor = -1.
        self.pref_list = None
        self.simmat_masked = None

        self.rand_state = None
        self.damping = None
        self.max_iter = None
        self.min_cluster_size = None

        self.results = None


    def __str__(self):
        return f"Clusterer(simmat_shape={self.simmat.shape})"


    def prepare_simmat(self):
        if not np.any(self.eligmat):
            raise ValueError("No eligible entries in eligmat")

        simmat_masked = np.where(self.eligmat, self.simmat, self.floor)

        mask = self.eligmat.copy()
        np.fill_diagonal(mask, False)

        positive = simmat_masked[mask]
        positive = positive[positive > 0]

        if positive.size == 0:
            raise ValueError("No positive similarities among eligible off-diagonal entries. Cannot compute pref_list")
        
        pref_list = np.round(np.percentile(positive, [5, 10, 25, 50, 75, 90, 95, 99]), decimals=5)

        return simmat_masked, pref_list
    

    def save_clustering_parameters(self, rand_state, damping, max_iter, min_cluster_size):
        self.rand_state = rand_state
        self.damping = damping
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size


    def prepare_clusterer(self, rand_state, damping, max_iter, min_cluster_size):
        self.save_clustering_parameters(rand_state, damping, max_iter, min_cluster_size)
        self.simmat_masked, self.pref_list = self.prepare_simmat()


    def cluster_simmat(self):
        self.results = run_ap_elig(self.simmat_masked, self.eligmat, self.rand_state, self.damping,
                                   self.max_iter, self.min_cluster_size, self.pref_list)
        return self.results


    def get_pref_list(self):
        return self.pref_list