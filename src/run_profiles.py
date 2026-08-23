import os
import shutil
import pickle
import argparse
import logging
import numpy as np
from datetime import datetime

# local library files
from logkeeper import setup_logging
from config import Config
from data_manager import get_path_data, prepare_data
from profiles import build_profiles, clean_profiles, add_sequences_cached, remove_low_freq_profiles
from io_manager import save_prof_dict_npz


def main():
    """
    Step 3: Profile creation and matching

    Saves:
    - raw profile dictionary [prof_dict]
    - profile dictionary without duplicates [clean_prof_dict]
    - clean profile dictionary with matched sequences added [seq_prof_dict]
    - final dictionary with only profiles that meet the minimum required frequency [final_prof_dict]
    """

    # prepares logger
    log_filename = setup_logging("run_profiles", logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("====== run_profiles: Start ======")

    # getting config file from CLI
    parser = argparse.ArgumentParser(description="Profile creation and matching")
    parser.add_argument("-c", "--config", default="config.json", help="Config file")
    args = parser.parse_args()
    config_filename = args.config

    # prepare config and print
    cfg = Config(config_filename)
    cfg.print_terminal()
    cfg.print_info()

    # get variables from config
    dataset, start_time_sec, duration_min = cfg.get_session_variables()
    max_rate_Hz, window_ms, slide_ms, bin_ms = cfg.get_binmat_variables()
    alpha, match_penalty, min_neurons = cfg.get_simmat_variables()
    spike_relevance_th, similarity_th, alignment_th = cfg.get_profile_variables()
    recall_th, min_hits, overlap_th = cfg.get_matching_variables()

    # get export directories from config
    suffix = cfg.expand_suffix_editsim(cfg.get_io_suffix())
    long_suffix = cfg.expand_suffix_clustering(suffix)
    prof_suffix = cfg.expand_suffix_profile(long_suffix)
    match_suffix = cfg.expand_suffix_matching(prof_suffix)
    export_dir, _ = cfg.get_output_directories(suffix)

    # prepare raw data
    logger.info("Preparing session data...")
    start = datetime.now()
    path_data = get_path_data(dataset)
    logger.info("Using data directory: %s", path_data)
    _, neuron_list, _, windows, _, _, end_elect_1 = prepare_data(path_data, start_time_sec,
                                                            duration_min, max_rate_Hz,
                                                            window_ms, slide_ms, bin_ms)
    logger.info(f"Neuron list (length {len(neuron_list)}): {neuron_list}")
    logger.info(f"Last neuron of electrode 1: {end_elect_1}")
    end = datetime.now()
    logger.info(f"Finished preparing session data. Elapsed time: {end-start}")

    # import clustering results
    with open(os.path.join(export_dir, "clustering_"+long_suffix+".pkl"), 'rb') as f:
        clustering_results = pickle.load(f)

    # create profile dictionary
    logger.info("Creating profiles...")
    start = datetime.now()
    prof_dict = build_profiles(windows, neuron_list, end_elect_1, 
                               clustering_results, alpha, 
                               spike_relevance_th, min_neurons)
    end = datetime.now()
    logger.info(f"Finished creating profiles. Elapsed time: {end-start}")

    # remove duplicates
    logger.info("Cleaning duplicate profiles...")
    start = datetime.now()
    clean_prof_dict = clean_profiles(prof_dict, alpha, similarity_th, alignment_th)
    end = datetime.now()
    logger.info(f"Finished cleaning duplicate profiles. Elapsed time: {end-start}")

    # add sequences (matching time windows) to dictionary
    logger.info("Adding sequences to profiles...")
    start = datetime.now()
    # seq_prof_dict = add_sequences(clean_prof_dict, windows, alpha, recall_th)
    seq_prof_dict = add_sequences_cached(clean_prof_dict, windows, alpha, recall_th, overlap_th)
    end = datetime.now()
    logger.info(f"Finished adding sequences. Elapsed time: {end-start}")

    # remove profiles that don't meet the minimum frequency required
    logger.info("Removing low frequency profiles...")
    start = datetime.now()
    final_prof_dict = remove_low_freq_profiles(seq_prof_dict, min_hits)
    end = datetime.now()
    logger.info(f"Finished removing low frequency profiles. Elapsed time: {end-start}")

    # print summary
    logger.info("Profile creation results")
    for key, item in final_prof_dict.items():
        size = item['size']
        ids = item['neurons']
        layer = item['layer']
        logger.info(f"-- Profile: {key} | {size} neurons: {ids} | Layer: {layer}")

    # export files
    logger.info("Exporting dictionaries...")
    start = datetime.now()
    with open(os.path.join(export_dir, "prof_dict_"+prof_suffix+".pkl"), 'wb') as f:
        pickle.dump(prof_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(export_dir, "clean_prof_dict_"+prof_suffix+".pkl"), 'wb') as f:
        pickle.dump(clean_prof_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(export_dir, "seq_prof_dict_"+match_suffix+".pkl"), 'wb') as f:
        pickle.dump(seq_prof_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(export_dir, "final_prof_dict_"+match_suffix+".pkl"), 'wb') as f:
        pickle.dump(final_prof_dict, f, pickle.HIGHEST_PROTOCOL)

    save_prof_dict_npz(final_prof_dict, os.path.join(export_dir, "final_prof_dict_"+match_suffix+".npz")) 

    end = datetime.now()
    logger.info(f"Finished exporting. Elapsed time: {end-start}")

    logger.info("====== run_profiles: Done ======")

    # copy logger
    logging.shutdown()
    shutil.copy2(log_filename, os.path.join(export_dir, os.path.basename(log_filename)))


if __name__ == "__main__":
    main()
