import numpy as np
from numba import njit


def edit_similarity(mat1, mat2, alpha=0.01, match_penalty=0, min_neurons=0):
    """
    !! DEPRECATED:
    Use methods _pure, _elig and _norm instead. Left here for future reference

    Calculates raw score, backpointer matrix, norm score by dice/F1
        ``2. * raw_score / (spkcount1 + spkcount2)`` 
    and eligibility with parameter min_neurons
    """
    # assert valid parameters
    assert alpha > 0, "alpha must be > 0 to produce meaningful gap penalties"
    assert match_penalty <= 0, "match_penalty must be <= 0 (penalty or neutral)"
    assert min_neurons >= 0, "min_neurons must be >= 0"

    tw1 = mat1.shape[1]
    tw2 = mat2.shape[1]

    # compute spike activity
    spkcount1 = float(mat1.sum())
    spkcount2 = float(mat2.sum())

    if (spkcount1 > 0) and (spkcount2 > 0):
        # pre-compute match matrix
        match = mat1.T @ mat2

        # pre-compute exponential penalties
        max_gap = max(tw1, tw2)
        exp_penalties = np.exp(alpha * np.arange(max_gap + 1)) - 1

        # call core function
        max_score, max_pos, eps, bp = _edit_similarity_core(
            match, exp_penalties, match_penalty, tw1, tw2
        )

        raw_score = max_score
        norm_score = 2. * raw_score / (spkcount1 + spkcount2)
    else:
        # no need to call core function: no spikes
        raw_score = 0.
        norm_score = 0.
        max_pos = (0, 0)
        eps = np.zeros((tw1 + 1, tw2 + 1), dtype=np.float32)
        bp = np.zeros((tw1 + 1, tw2 + 1), dtype=np.uint8)

    # eligibility: whether value will be considered for clustering
    elig = True
    if min_neurons > 0:
        nrncount1 = np.sum(np.any(mat1 > 0, axis=1))
        nrncount2 = np.sum(np.any(mat2 > 0, axis=1))

        # requires at least "elig_min" neurons active per window
        elig = (min(nrncount1, nrncount2) >= min_neurons)
    
    return raw_score, norm_score, max_pos, eps, bp, elig


def edit_similarity_pure(mat1, mat2, alpha=0.01, match_penalty=0):
    """
    Calculates raw score and backpointer matrix without eligibility
    """
    # assert valid parameters
    assert alpha > 0, "alpha must be > 0 to produce meaningful gap penalties"
    assert match_penalty <= 0, "match_penalty must be <= 0 (penalty or neutral)"

    tw1 = mat1.shape[1]
    tw2 = mat2.shape[1]

    # compute spike activity
    spkcount1 = float(mat1.sum())
    spkcount2 = float(mat2.sum())

    if (spkcount1 > 0) and (spkcount2 > 0):
        # pre-compute match matrix
        match = mat1.T @ mat2

        # pre-compute exponential penalties
        max_gap = max(tw1, tw2)
        exp_penalties = np.exp(alpha * np.arange(max_gap + 1)) - 1

        # call core function
        max_score, max_pos, eps, bp = _edit_similarity_core(
            match, exp_penalties, match_penalty, tw1, tw2
        )
    else:
        # no need to call core function: no spikes
        max_score = 0.
        max_pos = (0, 0)
        eps = np.zeros((tw1 + 1, tw2 + 1), dtype=np.float32)
        bp = np.zeros((tw1 + 1, tw2 + 1), dtype=np.uint8)
    
    return max_score, max_pos, eps, bp


def edit_similarity_elig(mat1, mat2, alpha=0.01, match_penalty=0, min_neurons=0, skip_low_activity=False):
    """
    Calculates raw score, backpointer matrix and eligibility with parameter min_neurons

    If skip_low_activity is set, optimizes calculation by skipping window pairs that don't
    meet the minimum eligibility requirement. In that case, eligibility is somewhat redundant,
    but may be used to identify windows that were skipped over windows with true zero similarity.
    """
    # assert valid parameters
    assert alpha > 0, "alpha must be > 0 to produce meaningful gap penalties"
    assert match_penalty <= 0, "match_penalty must be <= 0 (penalty or neutral)"
    assert min_neurons >= 0, "min_neurons must be >= 0"

    tw1 = mat1.shape[1]
    tw2 = mat2.shape[1]

    # # compute spike activity
    # spkcount1 = float(mat1.sum())
    # spkcount2 = float(mat2.sum())

    nrncount1 = np.sum(np.any(mat1 > 0, axis=1))
    nrncount2 = np.sum(np.any(mat2 > 0, axis=1))

    activity_threshold = min_neurons if skip_low_activity else 1

    if (nrncount1 >= activity_threshold) and (nrncount2 >= activity_threshold):
        # pre-compute match matrix
        match = mat1.T @ mat2

        # pre-compute exponential penalties
        max_gap = max(tw1, tw2)
        exp_penalties = np.exp(alpha * np.arange(max_gap + 1)) - 1

        # call core function
        max_score, max_pos, eps, bp = _edit_similarity_core(
            match, exp_penalties, match_penalty, tw1, tw2
        )
    else:
        # no need to call core function: no spikes
        max_score = 0.
        max_pos = (0, 0)
        eps = np.zeros((tw1 + 1, tw2 + 1), dtype=np.float32)
        bp = np.zeros((tw1 + 1, tw2 + 1), dtype=np.uint8)

    # eligibility: whether value will be considered for clustering
    elig = True
    if min_neurons > 0:
        # requires at least "elig_min" neurons active per window
        elig = (min(nrncount1, nrncount2) >= min_neurons)
    
    return max_score, max_pos, eps, bp, elig


def edit_similarity_norm(mat1, mat2, alpha=0.01, match_penalty=0):
    """
    Calculates raw score, backpointer matrix, eligibility with parameter
    min_neurons and norm score via the direct calculation of self-similarity
    """
    # assert valid parameters
    assert alpha > 0, "alpha must be > 0 to produce meaningful gap penalties"
    assert match_penalty <= 0, "match_penalty must be <= 0 (penalty or neutral)"

    tw1 = mat1.shape[1]
    tw2 = mat2.shape[1]

    # compute spike activity
    spkcount1 = float(mat1.sum())
    spkcount2 = float(mat2.sum())

    if (spkcount1 > 0) and (spkcount2 > 0):
        # pre-compute match matrix
        match = mat1.T @ mat2

        # self-similarity match matrices
        match_d1 = mat1.T @ mat1
        match_d2 = mat2.T @ mat2

        # pre-compute exponential penalties
        max_gap = max(tw1, tw2)
        exp_penalties = np.exp(alpha * np.arange(max_gap + 1)) - 1

        # call core function
        max_score, max_pos, eps, bp = _edit_similarity_core(
            match, exp_penalties, match_penalty, tw1, tw2
        )

        d1, _, _, _ = _edit_similarity_core(
            match_d1, exp_penalties[:tw1+1], match_penalty, tw1, tw1
        )

        d2, _, _, _ = _edit_similarity_core(
            match_d2, exp_penalties[:tw2+1], match_penalty, tw2, tw2
        )

        raw_score = max_score
        denom = np.sqrt(d1 * d2)
        norm_score = raw_score / denom if denom > 0 else 0.0 # just to be safe
    else:
        # no need to call core function: no spikes
        raw_score = 0.
        norm_score = 0.
        max_pos = (0, 0)
        eps = np.zeros((tw1 + 1, tw2 + 1), dtype=np.float32)
        bp = np.zeros((tw1 + 1, tw2 + 1), dtype=np.uint8)
    
    return raw_score, norm_score, max_pos, eps, bp


@njit(cache=True)
def _edit_similarity_core(match, exp_penalties, match_penalty, tw1, tw2):
    """
    Core of edit similarity calculation using Numba
    """
    eps = np.zeros((tw1 + 1, tw2 + 1), dtype=np.float32)
    bp = np.zeros((tw1 + 1, tw2 + 1), dtype=np.uint8)
    
    nu = np.zeros((tw1 + 1, tw2 + 1), dtype=np.int32) # vertical gap lengths
    rho = np.zeros((tw1 + 1, tw2 + 1), dtype=np.int32) # horizontal gap lengths
    
    max_score = 0.
    max_i = 0
    max_j = 0
    
    for i in range(1, tw1 + 1):
        for j in range(1, tw2 + 1):
            # diagonal (match/mismatch)
            match_local = match[i-1, j-1]
            match_local = match_penalty if match_local == 0 else match_local
            diag = eps[i-1, j-1] + match_local
            
            # vertical gap transition
            prev_nu = nu[i-1,j]
            start_vert = eps[i-1, j] - exp_penalties[1]
            
            if prev_nu > 0:
                extend_vert = eps[i - prev_nu - 1, j] - exp_penalties[prev_nu + 1]
            else:
                extend_vert = -np.inf
            
            if start_vert >= extend_vert:
                vert = start_vert
                nu_new = 1
            else:
                vert = extend_vert
                nu_new = prev_nu + 1
            
            # horizontal gap transition
            prev_rho = rho[i,j-1]
            start_horz = eps[i, j-1] - exp_penalties[1]
            
            if prev_rho > 0:
                extend_horz = eps[i, j - prev_rho - 1] - exp_penalties[prev_rho + 1]
            else:
                extend_horz = -np.inf
            
            if start_horz >= extend_horz:
                horz = start_horz
                rho_new = 1
            else:
                horz = extend_horz
                rho_new = prev_rho + 1
            
            # choose best operation
            best_score = 0.
            choice = 0
            
            if diag > best_score:
                best_score = diag
                choice = 1
            if vert > best_score:
                best_score = vert
                choice = 2
            if horz > best_score:
                best_score = horz
                choice = 3
            
            eps[i, j] = best_score
            bp[i, j] = choice
            
            # update gap lengths based on choice
            if choice == 1:  # diagonal
                nu[i, j] = 0
                rho[i, j] = 0
            elif choice == 2:  # vertical
                nu[i, j] = nu_new
                rho[i, j] = 0
            elif choice == 3:  # horizontal
                nu[i, j] = 0
                rho[i, j] = rho_new
            else:  # reset
                nu[i, j] = 0
                rho[i, j] = 0
            
            # track maximum
            if best_score > max_score:
                max_score = best_score
                max_i = i
                max_j = j

    return max_score, (max_i, max_j), eps, bp


def get_eligibility_pair(mat1, mat2, min_neurons):
    """
    Returns the eligibility of a pair of windows with respect to min_neurons
    """
    # assert valid parameters
    assert min_neurons >= 0, "min_neurons must be >= 0"

    # eligibility: whether this pair will be considered for clustering
    elig = True
    if min_neurons > 0:
        nrncount1 = np.sum(np.any(mat1 > 0, axis=1))
        nrncount2 = np.sum(np.any(mat2 > 0, axis=1))

        # requires at least "elig_min" neurons active per window
        elig = (min(nrncount1, nrncount2) >= min_neurons)
    
    return elig


def get_eligibility_single(mat, min_neurons):
    """
    Returns the eligibility of a single window with respect to min_neurons
    """
    # assert valid parameters
    assert min_neurons >= 0, "min_neurons must be >= 0"

    # eligibility: whether this window will be considered for clustering
    elig = True
    if min_neurons > 0:
        nrncount = np.sum(np.any(mat > 0, axis=1))

        # requires at least "elig_min" neurons active per window
        elig = (nrncount >= min_neurons)
    
    return elig