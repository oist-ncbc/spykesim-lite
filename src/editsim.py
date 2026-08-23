import os
import math
import numpy as np
from joblib import Parallel, delayed
from editsim_utils import edit_similarity_elig, edit_similarity_pure, get_eligibility_single


def _compute_tile_elig_symopt(i0, i1, j0, j1, windows, alpha, match_penalty, min_neurons, optz=False):
    """
    Computes one upper-triangle tile but only pairs with j >= i
    """
    # Pre-convert windows for this tile (keeps inner loop cheap)
    m_i = [windows[i].toarray() for i in range(i0, i1)]
    m_j = [windows[j].toarray() for j in range(j0, j1)]

    ii_list, jj_list, sim_list, elig_list = [], [], [], []

    for ii_off, i in enumerate(range(i0, i1)):
        m1 = m_i[ii_off]

        # enforce upper triangle inside tile
        j_start = max(j0, i)
        for j in range(j_start, j1):
            m2 = m_j[j - j0]

            max_score, _, _, _, elig = edit_similarity_elig(
                m1, m2, alpha, match_penalty, min_neurons, skip_low_activity=optz
            )

            ii_list.append(i)
            jj_list.append(j)
            sim_list.append(max_score)
            elig_list.append(elig)

    return (
        np.asarray(ii_list, dtype=np.int32),
        np.asarray(jj_list, dtype=np.int32),
        np.asarray(sim_list, dtype=np.float32),
        np.asarray(elig_list, dtype=bool),
    )


def calculate_editsim_elig_symopt(windows, alpha, match_penalty, min_neurons, optz=False):
    """
    Symmetric computation of edit similarity and eligibility using upper-triangle tiles
    """
    T = len(windows)
    n_cores = os.cpu_count() or 1

    n_side = max(1, int(math.ceil(4.0 * math.sqrt(n_cores))))
    tile_size = int(math.ceil(T / n_side))
    tile_size = max(128, min(1024, tile_size))

    # build upper-triangle tiles=
    jobs = []
    for i0 in range(0, T, tile_size):
        i1 = min(i0 + tile_size, T)
        for j0 in range(i0, T, tile_size):
            j1 = min(j0 + tile_size, T)
            jobs.append((i0, i1, j0, j1))

    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_compute_tile_elig_symopt)(i0, i1, j0, j1, windows, alpha, match_penalty, min_neurons, optz)
        for (i0, i1, j0, j1) in jobs
    )

    simmat = np.zeros((T, T), dtype=np.float32)
    eligmat = np.zeros((T, T), dtype=bool)

    # fill upper triangle from tiles then mirror
    for ii, jj, sims, eligs in results:
        simmat[ii, jj] = sims
        eligmat[ii, jj] = eligs

        simmat[jj, ii] = sims
        eligmat[jj, ii] = eligs

    return simmat, eligmat


def _compute_rows_elig(i_start, i_end, windows, T, alpha, match_penalty, min_neurons):
    """
    Computes edit similarity with eligibility for a block of the full similarity matrix
    """
    sim_block = np.empty((i_end - i_start, T), dtype=np.float32)
    elig_block = np.empty((i_end - i_start, T), dtype=bool)

    for ii, i in enumerate(range(i_start, i_end)):
        m1 = windows[i].toarray()
        for j in range(T):
            m2 = windows[j].toarray()

            max_score, _, _, _, elig = edit_similarity_elig(m1, m2, alpha, match_penalty, min_neurons)
            
            sim_block[ii, j] = max_score
            elig_block[ii, j] = elig

    return i_start, sim_block, elig_block


def calculate_editsim_elig(windows, alpha, match_penalty, min_neurons):
    """
    Oversees the calculation of edit similarity with eligibility using parallel jobs
    """
    # prepare jobs for simmat calculation
    T = len(windows)
    n_cores = os.cpu_count()
    block_size = max(T // (4 * n_cores), 50)
    jobs = [(i, min(i + block_size, T)) for i in range(0, T, block_size)]

    # parallel simmat calculation
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_compute_rows_elig)(i0, i1, windows, T, alpha, match_penalty, min_neurons)
        for i0, i1 in jobs
    )

    # save blocks into a single matrix
    simmat = np.zeros((T, T), dtype=np.float32)
    eligmat = np.zeros((T, T), dtype=bool)
    for i0, sim_block, elig_block in results:
        simmat[i0:i0 + sim_block.shape[0], :] = sim_block
        eligmat[i0:i0 + elig_block.shape[0], :] = elig_block

    return simmat, eligmat


def _compute_rows_pure(i_start, i_end, windows, T, alpha, match_penalty, min_neurons):
    """
    Computes edit similarity for a block of the full similarity matrix
    """
    sim_block = np.empty((i_end - i_start, T), dtype=np.float32)

    for ii, i in enumerate(range(i_start, i_end)):
        m1 = windows[i].toarray()
        for j in range(T):
            m2 = windows[j].toarray()

            max_score, _, _, _ = edit_similarity_pure(m1, m2, alpha, match_penalty, min_neurons)
            
            sim_block[ii, j] = max_score

    return i_start, sim_block


def calculate_editsim_pure(windows, alpha, match_penalty, min_neurons):
    """
    Oversees the calculation of edit similarity using parallel jobs
    """
    # prepare jobs for simmat calculation
    T = len(windows)
    n_cores = os.cpu_count()
    block_size = max(T // (4 * n_cores), 50)
    jobs = [(i, min(i + block_size, T)) for i in range(0, T, block_size)]

    # parallel simmat calculation
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_compute_rows_pure)(i0, i1, windows, T, alpha, match_penalty, min_neurons)
        for i0, i1 in jobs
    )

    # save blocks into a single matrix
    simmat = np.zeros((T, T), dtype=np.float32)
    for i0, sim_block in results:
        simmat[i0:i0 + sim_block.shape[0], :] = sim_block

    return simmat


def normalize_full_simmat(simmat):
    """
    Normalizes edit similarity matrix using
        norm_ij = sim_ij / sqrt(autosim_i * autosim_j)
    """
    simmat = np.asarray(simmat)
    diagonal = np.diag(simmat)
    diag_sqrt = np.sqrt(diagonal)

    denom = diag_sqrt[:, None] * diag_sqrt[None, :]

    norm_simmat = np.zeros_like(simmat, dtype=np.float32)
    np.divide(simmat, denom, out=norm_simmat, where=(denom != 0))
    
    return norm_simmat


def get_eligibility_vector(windows, min_neurons):
    """
    Returns the eligibility vector with respect to min_neurons

    Note: if eligmat has been calculated, extracting its diagonal should suffice
    """
    T = len(windows)
    windows_dense = [win.toarray() for win in windows]

    eligvec = np.zeros(T, dtype=bool)
    for i, win in enumerate(windows_dense):
        eligvec[i] = get_eligibility_single(win, min_neurons)

    return eligvec