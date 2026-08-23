import os
import shutil
import pickle
import logging
import argparse
import numpy as np

from datetime import datetime

# local library files
from logkeeper import setup_logging
from config import Config
from clusterer import Clusterer
from io_manager import save_clustering_results_npz


def run_clustering():
    """
    Step 2: Clustering with Affinity Propagation

    Saves:
    - clustering dictionary
    - list of preferences used
    """

    # prepares logger
    log_filename = setup_logging("run_clustering", logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("====== run_clustering: Start ======")

    # getting config file from CLI
    parser = argparse.ArgumentParser(description="Clustering with Affinity Propagation")
    parser.add_argument("-c", "--config", default="config.json", help="Config file")
    args = parser.parse_args()
    config_filename = args.config

    # prepare config and print
    cfg = Config(config_filename)
    cfg.print_terminal()
    cfg.print_info()

    # get variables from config
    rand_state, damping, max_iter, min_cluster_size = cfg.get_clustering_variables()

    # get export directories from config
    suffix = cfg.expand_suffix_editsim(cfg.get_io_suffix())
    long_suffix = cfg.expand_suffix_clustering(suffix)
    export_dir, _ = cfg.get_output_directories(suffix)

    # import calculated simmat and eligmat
    # simmat = np.load(os.path.join(export_dir, "simmat_"+suffix+".npy")) # raw simmat
    # norm_simmat = np.load(os.path.join(export_dir, "norm_simmat_"+suffix+".npy"))
    norm_simmat_symopt = np.load(os.path.join(export_dir, "norm_simmat_symopt_"+suffix+".npy"))
    eligmat = np.load(os.path.join(export_dir, "eligmat_"+suffix+".npy"))

    # create clustering assistant
    clstr = Clusterer(norm_simmat_symopt, eligmat)
    clstr.prepare_clusterer(rand_state, damping, max_iter, min_cluster_size)
    pref_list = clstr.get_pref_list()

    # run clustering and retrieve used preferences
    logger.info("Clustering simmat...")
    logger.info(f"Preference list: {pref_list}")
    start = datetime.now()
    clustering_results = clstr.cluster_simmat()
    end = datetime.now()
    logger.info(f"Finished clustering simmat. Elapsed time: {end-start}")

    # print summary
    logger.info("Clustering results")
    for key, item in clustering_results.items():
        n_clusters = item['n_clusters']
        n_valid = item['n_valid']
        logger.info(f"-- Preference: {key} | {n_clusters} clusters ({n_valid} valid)")
        converged = item['converged']
        if converged:
            valid_exemplars = item['valid_exemplars']
            valid_mem_count = item['valid_mem_count']
            logger.info(f"Valid exemplars: {valid_exemplars}")
            logger.info(f"Member count for valid clusters: {valid_mem_count}")
        else:
            logger.info("Didn't converge")

    # export clustering dictionary and preference list
    logger.info("Exporting clustering results and preference list...")
    start = datetime.now()

    with open(os.path.join(export_dir, "clustering_"+long_suffix+".pkl"), 'wb') as f:
        pickle.dump(clustering_results, f, pickle.HIGHEST_PROTOCOL)

    save_clustering_results_npz(clustering_results, os.path.join(export_dir, "clustering_"+long_suffix+".npz"))

    np.save(os.path.join(export_dir, "preflist_"+suffix+".npy"), pref_list)

    end = datetime.now()
    logger.info(f"Finished exporting. Elapsed time: {end-start}")

    logger.info("====== run_clustering: Done ======")

    # copy logger
    logging.shutdown()
    shutil.copy2(log_filename, os.path.join(export_dir, os.path.basename(log_filename)))


if __name__ == "__main__":
    run_clustering()