import os
import pandas as pd
import numpy as np

from datetime import datetime
from data_utils import import_spikes_etos4, extract_neurons_fr, clip_spike_train, remove_neurons, df2binarray_csc


def get_path_data(dataset: str):
    """
    Chooses directory to load data from depending on locality
    """
    data_root = "/Users/milena/DATA/raw_data/"
    path_data = os.path.join(data_root, dataset + "_new3/")

    abs_path_data = os.path.abspath(path_data)

    if not os.path.exists(abs_path_data):
        raise FileNotFoundError(f"Directory does not exist: {abs_path_data}")

    if not os.path.isdir(abs_path_data):
        raise NotADirectoryError(f"Path is not a directory: {abs_path_data}")

    return abs_path_data


def get_end_elect_1(df: pd.DataFrame):
    """
    Returns last neuron of first electrode for layer mapping
    """
    return df.loc[df.electrode == 1, "neuronid"].max()


def prepare_data(path_data: str, start_time_sec: float, duration_min: float, max_rate_Hz: float, window_ms: float, slide_ms: float, bin_ms: float, full: bool = True):
    """
    Loads ETOS4 spiking data using auxiliary functions from graph-check. 
    
    If "full," calculates the average firing rate of each neuron for the entire 
    session (default); if the firing rate calculation should be clipped to the 
    duration, set this variable as false
    """
    frequency_Hz = 20000
    start_time_ms = start_time_sec * 1000
    duration_ms = duration_min * 60 * 1000

    if full:
        df_spikes_temp = import_spikes_etos4(path_data, frequency_Hz)
        _, remove_neu = extract_neurons_fr(df_spikes_temp, max_rate_Hz)
        df_spikes_lowFR = remove_neurons(df_spikes_temp, remove_neu)
        df_spikes = clip_spike_train(df_spikes_lowFR, start_time_ms, duration_ms)
    else:
        df_spikes_temp = import_spikes_etos4(path_data, frequency_Hz)
        df_spikes_clip = clip_spike_train(df_spikes_temp, start_time_ms, duration_ms)
        _, remove_neu = extract_neurons_fr(df_spikes_clip, max_rate_Hz, duration_ms)
        df_spikes = remove_neurons(df_spikes_clip, remove_neu)
    
    end_elect_1 = get_end_elect_1(df_spikes)

    binmat, neuron_list = df2binarray_csc(df_spikes, start_time_ms, duration_ms, bin_ms)

    duration_bins = binmat.shape[1]
    window_len_bins = int(np.ceil(window_ms / bin_ms))
    slide_len_bins = int(np.ceil(slide_ms / bin_ms))

    start_bins = np.arange(0, duration_bins - window_len_bins + 1, slide_len_bins, dtype=int)
    windows = [binmat[:, t:(t + window_len_bins)] for t in start_bins]

    times_ms = start_time_ms + start_bins * bin_ms

    return binmat, neuron_list, start_bins, windows, times_ms, window_len_bins, end_elect_1