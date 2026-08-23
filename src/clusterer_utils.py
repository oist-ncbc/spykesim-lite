import numpy as np

from sklearn.cluster import AffinityPropagation as AP
from functools import partial
from joblib import Parallel, delayed


def ap_param_search_elig(pref, simmat, eligmat, rand_state, damping, max_iter, min_cluster_size):
    """
    Performs clustering using Affinity Propagation and investigates valid
    clusters using min_cluster_size and percentile-based eligible similarity
    comparison between members and non-members

    Current version assumes a symmetric similarity matrix
    """
    clusterer_ap = AP(
        preference=pref, affinity='precomputed', 
        random_state=int(rand_state), max_iter=max_iter,
        damping=damping
    )
    cluster_labels = np.asarray(clusterer_ap.fit_predict(simmat), dtype=np.int32)
    converged = bool(clusterer_ap.n_iter_ < max_iter)

    labels = np.unique(cluster_labels).astype(np.int32, copy=False)
    n_clusters = int(labels.size)

    n_valid_clu = 0
    valid_clu_list = []
    valid_clu_sim = []
    valid_clu_mem = []

    N = simmat.shape[0]
    all_idx = np.arange(N, dtype=np.int32)
    mask = np.zeros(N, dtype=bool)

    for clu in labels:
        clu = int(clu)
        clu_mem = np.where(cluster_labels == clu)[0].astype(np.int32, copy=False)
        k = int(clu_mem.size)

        if k < min_cluster_size:
            continue

        clu_indices = np.ix_(clu_mem, clu_mem)
        S_clu = simmat[clu_indices]
        E_clu = eligmat[clu_indices]
        
        triangle = np.triu(np.ones((k, k), dtype=bool), 1)
        within_vals = S_clu[E_clu & triangle].ravel()

        mask[:] = False
        mask[clu_mem] = True
        out_mem = all_idx[~mask]

        if out_mem.size == 0:
            continue

        out_indices = np.ix_(clu_mem, out_mem)
        S_out = simmat[out_indices]
        E_out = eligmat[out_indices]
        outside_vals = S_out[E_out].ravel()

        if within_vals.size == 0 or outside_vals.size == 0:
            continue

        if within_vals.size < 20 or outside_vals.size < 50:
            continue

        p25_in = np.percentile(within_vals, 25)
        p75_out = np.percentile(outside_vals, 75)

        sim_score = float(np.median(within_vals))
        different = (p25_in > p75_out)

        if different:
            n_valid_clu += 1
            valid_clu_list.append(clu)
            valid_clu_sim.append(sim_score)
            valid_clu_mem.append(k)

    pref_log = f"{pref:.5f}"

    valid_clu_array = np.asarray(valid_clu_list, dtype=np.int32)
    valid_sim_array = np.asarray(valid_clu_sim, dtype=np.float32)
    valid_mem_array = np.asarray(valid_clu_mem, dtype=np.int32)
    all_exemplars = np.asarray(clusterer_ap.cluster_centers_indices_, dtype=np.int32)
    
    label_to_ex = {int(l): int(e) for l, e in zip(labels.tolist(), all_exemplars.tolist())}
    valid_exemplars = np.asarray([label_to_ex[int(clu)] for clu in valid_clu_array], dtype=np.int32)
    valid_cluster_labels = np.where(
        np.isin(cluster_labels, valid_clu_array), cluster_labels, np.int32(-1),
    ).astype(np.int32, copy=False)

    return (pref_log, cluster_labels, n_clusters, all_exemplars,
            n_valid_clu, valid_clu_array, valid_cluster_labels, 
            valid_sim_array, valid_mem_array, valid_exemplars, converged)


def run_ap_elig(simmat, eligmat, rand_state, damping, max_iter, min_cluster_size, pref_list):
    """
    Asssists the parallelization of Affinity Propagation
    """
    num_pref = len(pref_list)

    ap_param_search_ = partial(
        ap_param_search_elig,
        simmat=simmat,
        eligmat=eligmat,
        damping=damping,
        max_iter=max_iter,
        min_cluster_size=min_cluster_size,
    )

    results = Parallel(n_jobs=num_pref)(
        delayed(ap_param_search_)(pref, rand_state=rand_state + idx)
        for idx, pref in enumerate(pref_list)
    )

    results_dict = {
        item[0]: {  # preference as key
            'all_labels': item[1], # labels for every time window
            'n_clusters': item[2], # number of clusters
            'exemplars': item[3], # exemplars for clusters found
            'n_valid': item[4], # number of valid clusters
            'valid_cluster_ids': item[5], # ids of clusters considered valid
            'valid_labels': item[6], # labels for every time window (-1 for invalid clusters)
            'valid_clu_sim': item[7], # representative similarity score of the cluster (current: median)
            'valid_mem_count': item[8], # number of members for valid clusters
            'valid_exemplars': item[9], # exemplars for valid clusters
            'converged': item[10] # whether preference converged
        }
        for item in results
    }

    return results_dict