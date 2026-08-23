import numpy as np
import logging
from itertools import combinations

from editsim_utils import edit_similarity_pure, edit_similarity_norm
from alignment import alignment_simple

logger = logging.getLogger(__name__)


def build_profiles(windows, neuron_list, end_elect_1, results_dict, 
                   alpha, spike_relevance_th, min_neurons):
    """
    Accumulates spikes of valid clusters in the exemplar time frame
    and keeps those which are relevant in the cluster as defined
    by spike_relevance_th. Creates profiles from valid clusters with
    at least min_neurons after thresholding
    """
    pref_list = list(results_dict.keys())
    prof_dict = {}

    windows_dense = [win.toarray() for win in windows]
    
    profile_idx = 0

    for pref in pref_list:
        logger.info(f"Processing preference: {pref}")

        profile_count = 0
        cluster_info = results_dict[pref]
        converged = bool(cluster_info["converged"])

        pref_log = str(pref)

        if not converged:
            logger.info("Didn't converge. Skipping")
        else:
            n_valid = int(cluster_info['n_valid'])
            valid_labels = np.asarray(cluster_info["valid_labels"], dtype=np.int32)
            valid_exemplars = np.asarray(cluster_info["valid_exemplars"], dtype=np.int32)
            valid_cluster_ids = np.asarray(cluster_info["valid_cluster_ids"], dtype=np.int32)

            for i in range(n_valid):
                cluster_idx = int(valid_cluster_ids[i])
                exemplar_idx = int(valid_exemplars[i])
                exemplar = windows_dense[exemplar_idx]

                profile_temp = np.zeros_like(exemplar, dtype=np.int32)
                n_windows = 0

                cluster_members = np.where(valid_labels == cluster_idx)[0].astype(np.int32, copy=False)

                for member_idx in cluster_members:
                    # skip the exemplar using the lines below
                    # if member_idx == exemplar_idx:
                    #     continue

                    win = windows_dense[int(member_idx)]

                    _, max_pos, _, bp = edit_similarity_pure(exemplar, win, alpha=alpha)
                    al1, _ = alignment_simple(bp, max_pos, exemplar, win, mode='switch')

                    profile_temp += al1
                    n_windows += 1

                threshold = n_windows * spike_relevance_th
                profile_temp[profile_temp < threshold] = 0

                n_active_neurons = int(np.sum(np.any(profile_temp > 0, axis=1)))

                if n_active_neurons >= int(min_neurons):
                    profile = (profile_temp > 0).astype(np.uint8)

                    n_spikes = int(np.count_nonzero(profile))
                    is_active = np.any(profile > 0, axis=1)
                    active_neuron_ids = np.nonzero(is_active)[0].astype(np.int32, copy=False)

                    org_neuron_ids = np.asarray([int(neuron_list[int(i)]) for i in active_neuron_ids], dtype=np.int32)
                    org_neuron_elect = np.asarray([2 if idx > end_elect_1 else 1 for idx in org_neuron_ids], dtype=np.uint8)
                    
                    all_elect_vals = set(org_neuron_elect)
                    layer = 1 if all_elect_vals == {1} else 2 if all_elect_vals == {2} else 0
                    layer = int(layer)

                    prof_dict[int(profile_idx)] = {
                        "pref": pref_log,
                        "profile": profile,
                        "size": n_active_neurons,
                        "spkcount": n_spikes,
                        "neuron_idx": active_neuron_ids,
                        "neurons": org_neuron_ids,
                        "electrode": org_neuron_elect,
                        "layer": layer, # 0: cross-layer, 1: only electrode 1, 2: only electrode 2
                        "ref_cluster": cluster_idx,
                        "ref_exemplar": exemplar,
                        "ref_exemplar_idx": exemplar_idx,
                        "ref_members_idx": cluster_members
                    }

                    profile_idx += 1
                    profile_count += 1

            logger.info(f"Preference {pref}: {profile_count} profiles")

    return prof_dict


def clean_profiles(prof_dict, alpha, similarity_th, alignment_th):
    """
    Checks for duplicate profiles both in alignment and similarity
    and, for each duplicate group, keeps the profile with the lowest
    spike count
    """
    keys = [int(k) for k in sorted(prof_dict.keys())]
    duplicate_dict = {}

    for key1, key2 in combinations(keys, 2):
        prof1 = prof_dict[key1]["profile"]
        prof2 = prof_dict[key2]["profile"]

        _, norm_score, max_pos, _, bp = edit_similarity_norm(prof1, prof2, alpha=alpha)
        al1, al2 = alignment_simple(bp, max_pos, prof1, prof2, mode="multiply")

        spkcount1 = int(np.count_nonzero(prof1))
        spkcount2 = int(np.count_nonzero(prof2))

        alignspk1 = int(np.count_nonzero(al1))
        alignspk2 = int(np.count_nonzero(al2))

        frac1 = (alignspk1 / spkcount1) if spkcount1 else 0.0
        frac2 = (alignspk2 / spkcount2) if spkcount2 else 0.0

        encapsulated = (frac1 >= alignment_th - 1e-9) and (frac2 >= alignment_th - 1e-9)
        similar = (norm_score >= similarity_th - 1e-6)

        if encapsulated or similar:
            duplicate_dict.setdefault(key1, []).append(key2)
            duplicate_dict.setdefault(key2, []).append(key1)

    def spike_count_profile(k):
        prof = prof_dict[k]["profile"]
        return int(np.count_nonzero(prof))
    
    keep = set()
    seen = set()
    rep_to_group = {}

    for start_key in duplicate_dict:
        if start_key in seen:
            continue

        duplicate_group = set()
        stack = [start_key]

        while stack:
            k = stack.pop()
            if k in duplicate_group:
                continue
            duplicate_group.add(k)
            for n in duplicate_dict.get(k, []):
                if n not in duplicate_group:
                    stack.append(n)

        choice_prof = min(duplicate_group, key=lambda k: (spike_count_profile(k), k))
        keep.add(int(choice_prof))
        rep_to_group[int(choice_prof)] = {int(x) for x in duplicate_group}
        seen |= duplicate_group

    keep |= (set(keys) - set(duplicate_dict.keys()))

    clean_prof_dict = {}
    for k in keep:
        k_int = int(k)
        entry = prof_dict[k_int].copy()
        dupes = rep_to_group.get(k_int, set()) - {k_int}
        entry["duplicates"] = np.asarray([int(x) for x in sorted(dupes)], dtype=np.int32)
        clean_prof_dict[k_int] = entry

        if dupes:
            logger.info(f"Removed {len(dupes)} duplicates of profile {k_int}: {dupes}")
        else:
            logger.info(f"No duplicates to remove for profile {k_int}")

    clean_prof_dict = {int(k): clean_prof_dict[int(k)] for k in sorted(clean_prof_dict)}

    return clean_prof_dict


def add_sequences(prof_dict, windows, alpha, recall_th):
    """
    Adds sequences to profile dictionary if recall is higher than recall_th

    Does not check for overlap in candidate profiles. If necessary, 
    use add_sequences_cached instead
    """
    updated_prof_dict = {}

    windows_dense = [win.toarray() for win in windows]

    for k, item in prof_dict.items():
        k_int = int(k)
        prof = item["profile"]
        spk_count_prof = int(np.count_nonzero(prof))

        hit_idx = []
        hit_coverage = []
        hit_coverage_raw = []
        hit_recall = []
        hit_match_windows = []

        for j, win in enumerate(windows_dense):
            raw_score, norm_score, max_pos, _, bp = edit_similarity_norm(prof, win, alpha=alpha)
            _, al2 = alignment_simple(bp, max_pos, prof, win, mode="multiply")

            recall = (int(np.count_nonzero(al2)) / spk_count_prof) if spk_count_prof else 0.0

            if recall >= recall_th - 1e-9:
                hit_idx.append(int(j))
                hit_coverage.append(norm_score)
                hit_coverage_raw.append(raw_score)
                hit_recall.append(float(recall))
                hit_match_windows.append((2 * al2 - win).astype(np.int8, copy=False))

        logger.info(f"Profile {k_int}: {len(hit_idx)} hits")

        entry = item.copy()

        entry["sequences"] = hit_idx
        entry["coverage"] = np.array(hit_coverage, dtype=np.float32)
        entry["coverage_raw"] = np.array(hit_coverage_raw, dtype=np.float32)
        entry["recall"] = np.array(hit_recall, dtype=np.float32)
        entry["match_windows"] = hit_match_windows

        updated_prof_dict[k_int] = entry

    return updated_prof_dict


def add_sequences_cached(prof_dict, windows, alpha, recall_th, overlap_th):
    """
    Investigates candidate profiles for every window using recall_th and saves:
        (1) Those that do not overlap with other profiles
        (2) Those which do by at least overlap_th spikes but have 
        the highest values of norm. edit similarity
    """
    def spike_count_window(x):
        return int(np.count_nonzero(x))

    prof_keys = [int(k) for k in sorted(prof_dict.keys())]
    n_profiles = len(prof_keys)
    T = len(windows)

    windows_dense = [win.toarray() for win in windows]

    overlap_cache = {}

    def profile_overlap_check(i1, i2):
        """
        Indexing profiles in an orderly manner, saves a cache dictionary
        with overlap information for every profile pair

        Redundancy is identified by profiles sharing at least overlap_th spikes
        """
        if i1 == i2:
            return True
        if i1 > i2:
            i1, i2 = i2, i1

        key = (i1, i2)
        cached = overlap_cache.get(key, None)
        if cached is not None:
            return cached

        prof1 = prof_dict[prof_keys[i1]]["profile"]
        prof2 = prof_dict[prof_keys[i2]]["profile"]

        _, _, max_pos, _, bp = edit_similarity_norm(prof1, prof2, alpha=alpha)
        al1, al2 = alignment_simple(bp, max_pos, prof1, prof2, mode="multiply")

        redundant = (spike_count_window(al1) >= overlap_th) and (spike_count_window(al2) >= overlap_th)
        overlap_cache[key] = redundant

        return redundant

    # (1): collect candidate profiles for all windows checking for good recall
    candidates = [[] for _ in range(T)]

    for i, k in enumerate(prof_keys):
        prof = prof_dict[k]["profile"]
        spk_count_prof = spike_count_window(prof)

        for j, win in enumerate(windows_dense):
            raw_score, norm_score, max_pos, _, bp = edit_similarity_norm(prof, win, alpha=alpha)
            _, al2 = alignment_simple(bp, max_pos, prof, win, mode="multiply")

            recall = (spike_count_window(al2) / spk_count_prof) if spk_count_prof else 0.0
            if recall < recall_th - 1e-9:
                continue

            match_window = (2 * al2 - win).astype(np.int8, copy=False)
            candidates[j].append((int(i), norm_score, raw_score, float(recall), match_window))

    hit_idx = [[] for _ in range(n_profiles)]
    hit_cov = [[] for _ in range(n_profiles)]
    hit_cov_raw = [[] for _ in range(n_profiles)]
    hit_recall = [[] for _ in range(n_profiles)]
    hit_match_windows = [[] for _ in range(n_profiles)]

    # (2) resolve candidates per window using the cache dictionary
    for j, cands in enumerate(candidates):
        if not cands:
            continue

        m = len(cands)
        adj = [set() for _ in range(m)]

        # build redundancy graph between candidates
        for a, b in combinations(range(m), 2):
            i1, _, _, _, _ = cands[a]
            i2, _, _, _, _ = cands[b]

            if profile_overlap_check(i1, i2):
                adj[a].add(b)
                adj[b].add(a)

        # find connected components (redundant groups)
        seen = set()
        keep = set()

        for start in range(m):
            if start in seen:
                continue

            stack = [start]
            seen.add(start)
            comp = []

            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)

            # keep best-scoring candidate(s) in this group
            best = max(cands[u][1] for u in comp) # norm_score (float32)
            for u in comp:
                if best - cands[u][1] <= 1e-6:
                    keep.add(u)

        # assign window to kept candidates
        for u in keep:
            i, norm_s, raw_s, rec, mw = cands[u]
            hit_idx[i].append(j)
            hit_cov[i].append(norm_s)
            hit_cov_raw[i].append(raw_s)
            hit_recall[i].append(rec)
            hit_match_windows[i].append(mw)

    # (3) write results back
    updated_prof_dict = {}
    for i, k in enumerate(prof_keys):
        entry = prof_dict[k].copy()

        entry["sequences"] = np.asarray(hit_idx[i], dtype=np.int32)
        entry["coverage"] = np.asarray(hit_cov[i], dtype=np.float32)
        entry["coverage_raw"] = np.asarray(hit_cov_raw[i], dtype=np.float32)
        entry["recall"] = np.asarray(hit_recall[i], dtype=np.float32)
        entry["match_windows"] = hit_match_windows[i]
        updated_prof_dict[k] = entry

        logger.info(f"Profile {k}: {len(hit_idx[i])} hits")

    return updated_prof_dict


def remove_low_freq_profiles(prof_dict, min_hits):
    """
    Keeps only profiles with at least min_hits
    """
    return {
        k: item
        for k, item in prof_dict.items()
        if len(item["sequences"]) >= min_hits
    }