import numpy as np


def save_clustering_results_npz(results_dict, filename):
    """
    Saves clustering results dictionary into one .npz file

    NPZ keys:
        - "__prefs__" -> list of preference keys (already converted to strings)
        - "pref=<pref_str>/<field>" -> numpy array (scalars become 0-D arrays)
    """

    pack = {}

    # keys are now strings: change this if going back to floats
    pref_strs = list(results_dict.keys())
    pack["__prefs__"] = np.asarray(pref_strs, dtype=str)

    for pref_str, entry in results_dict.items():
        pref_str = str(pref_str) # plain python for stable key naming

        for field, value in entry.items():
            pack[f"pref={pref_str}/{field}"] = np.asarray(value)

    np.savez_compressed(filename, **pack)


def load_clustering_results_npz(filename):
    """
    Loads an NPZ created by save_clustering_results_npz

    Notes:
        - Preferences are stored and returned as strings (dict keys stay str)
        - 0-D numpy arrays are converted back to Python scalars via .item()
    """

    z = np.load(filename, allow_pickle=False)

    pref_strs = z["__prefs__"]
    results_dict = {}

    for pref_str in pref_strs:
        pref_str = str(pref_str) # numpy.str_ -> str
        entry = {}

        prefix = f"pref={pref_str}/"

        for name in z.files:
            if not name.startswith(prefix):
                continue

            field = name[len(prefix):]
            arr = z[name]

            entry[field] = arr.item() if arr.ndim == 0 else arr

        # keys remain strings now
        results_dict[pref_str] = entry

    return results_dict


def save_prof_dict_npz(prof_dict, filename):
    """
    Saves profile dictionary into one .npz file

    Notes:
        - Scalars become 0-D arrays
        - `match_windows` (list of 2D arrays) is stored as many separate arrays
    """

    pack = {}
    pack["__keys__"] = np.array(list(prof_dict.keys()), dtype=np.int64)

    for pid, entry in prof_dict.items():
        pid = int(pid)

        for k, v in entry.items():

            # match_windows is a list of 2D arrays
            # npz cannot store a list directly, so we store:
            #   - how many windows there are
            #   - each window as its own array
            if k == "match_windows":
                pack[f"{pid}/{k}/n"] = np.array(len(v), dtype=np.int64)

                for i, arr in enumerate(v):
                    # each window becomes its own dataset
                    pack[f"{pid}/{k}/{i}"] = np.asarray(arr, dtype=np.int8)

                continue

            # scalars / lists / arrays all become numpy arrays
            pack[f"{pid}/{k}"] = np.asarray(v)

    np.savez_compressed(filename, **pack)


def load_prof_dict_npz(filename):
    """
    Load a prof_dict saved with save_prof_dict_npz()

    Notes:
        - 0-D arrays are converted back to Python scalars
        - match_windows is reconstructed as a list of 2D arrays
    """

    z = np.load(filename, allow_pickle=False)
    keys = z["__keys__"].astype(np.int64)

    prof_dict = {}

    for pid in keys:
        pid = int(pid)
        entry = {}

        prefix = f"{pid}/"

        for name in z.files:
            if not name.startswith(prefix):
                continue

            if name.startswith(f"{pid}/match_windows/"):
                continue

            field = name[len(prefix):]
            arr = z[name]

            if arr.ndim == 0:
                entry[field] = arr.item()
            else:
                entry[field] = arr

        # reconstruct list of 2D arrays
        nname = f"{pid}/match_windows/n"
        if nname in z:
            n = int(z[nname].item())
            entry["match_windows"] = [
                z[f"{pid}/match_windows/{i}"].copy() for i in range(n)
            ]

        prof_dict[pid] = entry

    return prof_dict
