import os
import shutil
import logging
import argparse
import numpy as np

from datetime import datetime
from scipy.sparse import save_npz

# local library files
from logkeeper import setup_logging
from config import Config
from data_manager import get_path_data, prepare_data
from editsim import normalize_full_simmat, calculate_editsim_elig_symopt
from plotting import plot_simmat, plot_simmat_vs_norm, plot_eligmat, plot_firing_rate


def run_simmat():
    """
    Step 1: Edit similarity matrix calculation

    Saves:
    - binmat file (binned spike trains)
    - simmat file (edit similarity matrix with raw score)
    - norm_simmat file (edit similarity matrix with normalized score)
    - elig file (boolean matrix based on window activity for clustering)
    """

    # prepares logger
    log_filename = setup_logging("run_simmat", logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("====== run_simmat: Start ======")

    # getting config file from CLI
    parser = argparse.ArgumentParser(description="Edit similarity matrix calculation")
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

    # get export directories from config
    suffix = cfg.expand_suffix_editsim(cfg.get_io_suffix())
    export_dir, export_fig_dir = cfg.get_output_directories(suffix)

    # prepare raw data
    logger.info("Preparing session data...")
    start = datetime.now()
    path_data = get_path_data(dataset)
    logger.info("Using data directory: %s", path_data)
    _, neuron_list, _, windows, times_ms, _, end_elect_1 = prepare_data(path_data, start_time_sec,
                                                            duration_min, max_rate_Hz,
                                                            window_ms, slide_ms, bin_ms)
    logger.info(f"Neuron list (length {len(neuron_list)}): {neuron_list}")
    logger.info(f"Last neuron of electrode 1: {end_elect_1}")
    end = datetime.now()
    logger.info(f"Finished preparing session data. Elapsed time: {end-start}")

    # calculate simmat and eligmat
    logger.info("Starting simmat calculation with eligibility (symmetric approximation, optimized)...")
    start = datetime.now()
    simmat_symopt, eligmat = calculate_editsim_elig_symopt(windows, alpha, match_penalty, min_neurons, optz=True)
    end = datetime.now()
    logger.info(f"Finished simmat calculation with eligibility. Elapsed time: {end-start}")

    # normalize simmat
    logger.info("Normalizing simmat...")
    start = datetime.now()
    norm_simmat_symopt = normalize_full_simmat(simmat_symopt)
    end = datetime.now()
    logger.info(f"Finished simmat normalization. Elapsed time: {end-start}")

    # new: calculate eligibility vector (whether window meets the min_neurons requirement)
    # eligvec = get_eligibility_vector(windows, min_neurons)

    # sanity test: print percentiles
    logger.info("Simmat percentiles (raw/norm, symmetric approximation, optimized):")
    logger.info(f"P50: {np.percentile(simmat_symopt, 50)} {np.percentile(norm_simmat_symopt, 50)}")
    logger.info(f"P75: {np.percentile(simmat_symopt, 75)} {np.percentile(norm_simmat_symopt, 75)}")
    logger.info(f"P90: {np.percentile(simmat_symopt, 90)} {np.percentile(norm_simmat_symopt, 90)}")
    logger.info(f"P95: {np.percentile(simmat_symopt, 95)} {np.percentile(norm_simmat_symopt, 95)}")
    logger.info(f"P99: {np.percentile(simmat_symopt, 99)} {np.percentile(norm_simmat_symopt, 99)}") 

    # export files
    logger.info("Exporting binmat and simmat...")
    start = datetime.now()
    logger.info("Using export directory: %s", export_dir)

    # export binmat
    # save_npz(os.path.join(export_dir, "binmat_"+suffix+".npz"), binmat)

    # export raw simmat: symmetric and optimized
    # np.save(os.path.join(export_dir, "simmat_symopt_"+suffix+".npy"), simmat_symopt)

    # export norm simmat: symmetric and optimized
    np.save(os.path.join(export_dir, "norm_simmat_symopt_"+suffix+".npy"), norm_simmat_symopt)

    # export eligmat
    np.save(os.path.join(export_dir, "eligmat_"+suffix+".npy"), eligmat)

    # export eligvec
    # np.save(os.path.join(export_dir, "eligvec_"+suffix+".npy"), eligvec)

    shutil.copy2(config_filename, os.path.join(export_dir, os.path.basename(config_filename)))
    end = datetime.now()
    logger.info(f"Finished exporting. Elapsed time: {end-start}")

    # plot simmat visualization
    logger.info("Plotting simmat visualization...")
    start = datetime.now()
    logger.info("Using export directory: %s", export_fig_dir)
    plot_simmat(simmat_symopt, export_fig_dir, suffix)
    plot_simmat_vs_norm(simmat_symopt, norm_simmat_symopt, export_fig_dir, suffix)
    plot_eligmat(eligmat, norm_simmat_symopt, export_fig_dir, suffix)
    plot_firing_rate(windows, window_ms, neuron_list, times_ms, end_elect_1, export_fig_dir, suffix)
    end = datetime.now()
    logger.info(f"Finished plotting. Elapsed time: {end-start}")

    logger.info("====== run_simmat: Done ======")
    
    # copy logger
    logging.shutdown()
    shutil.copy2(log_filename, os.path.join(export_dir, os.path.basename(log_filename)))


if __name__ == "__main__":
    run_simmat()
