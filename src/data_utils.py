import os
import pandas as pd
import numpy as np
from scipy import sparse


def import_spikes_etos4(file_path: str, frequency_Hz: float):
    """
    Import spikes sorted using ETOS4 using files *.clu and *.res
    """
    prefix = 'All'
    file_type = ['clu','res']
    index_list = [1, 2]

    df_spikes = pd.DataFrame()
    id_shift = 0

    for idx in index_list:
        clu_file = f"{prefix}.{file_type[0]}.{idx}"
        res_file = f"{prefix}.{file_type[1]}.{idx}"
        
        classid = np.loadtxt(os.path.join(file_path, clu_file), dtype=np.int32)
        spiketime = np.loadtxt(os.path.join(file_path, res_file), dtype=np.float64)
        
        # ignore cluster count (first line of the clu file)
        classid = classid[1:]

        # convert to ms using frequency
        spiketime = spiketime[classid > 0] / frequency_Hz * 1000.0

        # remove cluster 0 ('outliers') from data
        classid = classid[classid > 0]
        
        df_seq = pd.DataFrame({
            'neuronid': classid + id_shift,
            'spiketime': spiketime,
            'electrode': [idx] * len(spiketime)
            }).astype({'neuronid': np.int32, 'spiketime': np.float64, 'electrode': np.uint8})

        # compute ID shift after current electrode
        unique_clu = np.unique(classid).astype(int)
        id_shift = id_shift + unique_clu.shape[0]

        df_spikes = pd.concat([df_spikes, df_seq], ignore_index=True, sort=False)

    assert df_spikes.groupby("neuronid")["electrode"].nunique().max() <= 1, "Duplicate neuron IDs detected across electrodes"

    ids = df_spikes["neuronid"].dropna().sort_values().unique()
    assert (ids == range(ids.min(), ids.max() + 1)).all(), "Neuron ID sequence has gaps"

    # shift first neuron id to 0
    df_spikes["neuronid"] = df_spikes["neuronid"] - df_spikes["neuronid"].min()

    df_spikes = df_spikes.reset_index(drop=True)

    return df_spikes


def clip_spike_train(df_spikes: pd.DataFrame, start_time_ms: float, duration_ms: float):
    end_time_ms = start_time_ms + duration_ms
    return df_spikes.loc[(df_spikes.spiketime >= start_time_ms) & (df_spikes.spiketime < end_time_ms)]


def extract_neurons_fr(df_spikes: pd.DataFrame, max_rate_Hz: float, duration_ms: float = None):
    rates = []
    remove_neu = []
    remove_flag = []

    if not duration_ms:
        duration_ms = df_spikes["spiketime"].max()

    neu_list = df_spikes["neuronid"].unique()
    
    remove_flag = np.zeros_like(neu_list, dtype=bool)

    for i, neu in enumerate(neu_list):
        df_temp = df_spikes.loc[df_spikes.neuronid == neu]

        rate_neuron = df_temp.shape[0] / (duration_ms) * 1000
        rates.append(rate_neuron)

        if (rate_neuron >= max_rate_Hz):
            remove_flag[i] = True
            remove_neu.append(neu)
            
    df_fr = pd.DataFrame({'neuronid': neu_list, 'rate': rates, 'remove': remove_flag})
            
    return df_fr, remove_neu


def remove_neurons(df_spikes: pd.DataFrame, remove_neu: list):
    index_list = list(df_spikes.loc[df_spikes.neuronid.isin(remove_neu)].index)
    df_spikes = df_spikes.drop(index=index_list)

    return df_spikes


def df2binarray_csc(df_spikes: pd.DataFrame, start_time_ms: float, duration_ms: float, bin_ms: float):
    """
    Transforms a spike dataframe into a sparse matrix of binned spike counts.

    - df_spikes must be pre-clipped to the half-open interval [start_time_ms, start_time_ms + duration_ms)
    - spike times, start_time_ms, duration_ms and bin_ms may be floating-point
    - bins are left-closed/right-open intervals of width bin_ms starting at start_time_ms
    """

    assert df_spikes.spiketime.min() >= start_time_ms, "First spike time happens before specified stage"
    assert df_spikes.spiketime.max() < start_time_ms + duration_ms, "Last spike time happens after specified stage"

    unique_ids = np.sort(df_spikes.neuronid.unique())
    neuron_list = [int(x) for x in unique_ids]

    neuronids = df_spikes.neuronid
    spikes_ms = df_spikes.spiketime # spike time already in ms
    
    nrow = len(unique_ids)
    ncol = int(np.ceil(duration_ms / bin_ms))
    bins = start_time_ms + np.arange(ncol) * bin_ms

    binarray_lil = sparse.lil_matrix((nrow, ncol), dtype=np.uint8)

    for i, n_id in enumerate(unique_ids):
        spk_train = spikes_ms[neuronids == n_id]
        digitized_spk_train = np.digitize(spk_train, bins) - 1
        binned_spk_train = np.bincount(digitized_spk_train)
        binarray_lil[i, digitized_spk_train] = binned_spk_train[digitized_spk_train]
    
    return binarray_lil.tocsc(), neuron_list