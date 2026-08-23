import os
import json
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


REQUIRED_CONFIG_KEYS = {
    "dataset",
    "start_time_sec",
    "duration_min",
    "max_rate_Hz",
    "window_ms",
    "slide_ms",
    "bin_ms",
    "alpha",
    "match_penalty",
    "min_neurons",
    "rand_state",
    "damping",
    "max_iter",
    "min_cluster_size",
    "spike_relevance_th",
    "similarity_th",
    "alignment_th",
    "recall_th",
    "min_hits",
    "overlap_th"
}


class Config:
    def __init__(self, config_file: str):
        self.config_file = config_file

        try:
            with open(config_file, "r") as f:
                self.config = json.load(f)

            self._validate_required_keys()

        except FileNotFoundError:
            logger.exception("Config file not found: %s", config_file)
            raise

        except json.JSONDecodeError as e:
            logger.exception("Invalid JSON in %s", config_file)
            raise ValueError(f"Invalid JSON in {config_file}") from e

        except ValueError as e:
            logger.exception("Config validation failed: %s (%s)", config_file, e)
            raise


    def _validate_required_keys(self) -> None:
        missing = REQUIRED_CONFIG_KEYS - self.config.keys()
        if missing:
            raise ValueError(f"Config missing required keys: {', '.join(sorted(missing))}")
        
        extra = self.config.keys() - REQUIRED_CONFIG_KEYS
        if extra:
            raise ValueError(f"Config has unknown keys: {', '.join(sorted(extra))}")
        
        if self["duration_min"] <= 0:
            raise ValueError("duration_min must be > 0")


    def __str__(self):
        return f"Config(config_file={self.config_file})"
    

    def __getitem__(self, key):
        try:
            return self.config[key]
        except KeyError:
            logger.error("Key '%s' has not been configured", key)
            raise


    def get_io_suffix(self):
        suffix = (
            f"dataset{self['dataset']}"
            f"_start{self['start_time_sec']}"
            f"_dur{self['duration_min']}"
            f"_fr{self['max_rate_Hz']}"
            f"_win{self['window_ms']}"
            f"_slide{self['slide_ms']}"
            f"_bin{self['bin_ms']}"
        )
        return suffix


    def expand_suffix_editsim(self, suffix):
        return (
            f"{suffix}"
            f"_alpha{self['alpha']}"
            f"_matchpen{self['match_penalty']}"
            f"_minneu{self['min_neurons']}"
        )


    def expand_suffix_clustering(self, suffix):
        return (
            f"{suffix}"
            f"_rand{self['rand_state']}"
            f"_damp{self['damping']}"
            f"_maxiter{self['max_iter']}"
            f"_minclu{self['min_cluster_size']}"
        )


    def expand_suffix_profile(self, suffix):
        return (
            f"{suffix}"
            f"_relev{self['spike_relevance_th']}"
            f"_sim{self['similarity_th']}"
            f"_align{self['alignment_th']}"
        )


    def expand_suffix_matching(self, suffix):
        return (
            f"{suffix}"
            f"_recall{self['recall_th']}"
            f"_minhits{self['min_hits']}"
            f"_overlap{self['overlap_th']}"
        )


    def get_output_directories(self, suffix):
        export_dir = f"./output/{suffix}/"
        export_dir_full = os.path.abspath(export_dir)
        os.makedirs(export_dir_full, exist_ok=True)

        export_fig_dir = f"{export_dir}images/"
        export_fig_dir_full = os.path.abspath(export_fig_dir)
        os.makedirs(export_fig_dir_full, exist_ok=True)
        return export_dir_full, export_fig_dir_full


    def get_session_variables(self):
        return self["dataset"], self["start_time_sec"], self["duration_min"]


    def get_binmat_variables(self):
        return self["max_rate_Hz"], self["window_ms"], self["slide_ms"], self["bin_ms"]


    def get_simmat_variables(self):
        return self["alpha"], self["match_penalty"], self["min_neurons"]


    def get_clustering_variables(self):
        return self["rand_state"], self["damping"], self["max_iter"], self["min_cluster_size"]


    def get_profile_variables(self):
        return self["spike_relevance_th"], self["similarity_th"], self["alignment_th"]


    def get_matching_variables(self):
        return self["recall_th"], self["min_hits"], self["overlap_th"]
    

    def _lines(self):
        return [
            "=== CONFIGURATION ===",
            "(A) Session and stage",
            f"Dataset: {self['dataset']} | Start (s): {self['start_time_sec']}  | Duration (min): {self['duration_min']}",
            f"Max. firing rate (Hz): {self['max_rate_Hz']} | Window (ms): {self['window_ms']} | Slide (ms): {self['slide_ms']} | Bin (ms): {self['bin_ms']}",
            "(B) Edit similarity",
            f"Alpha: {self['alpha']} | Match penalty: {self['match_penalty']} | Min. neurons: {self['min_neurons']}",
            "(C) Clustering",
            f"Random state: {self['rand_state']} | Damping: {self['damping']} | Max. iterations: {self['max_iter']} | Min. cluster size: {self['min_cluster_size']}",
            "(D) Profile creation",
            f"Spike relevance threshold: {self['spike_relevance_th']} | Sim. threshold: {self['similarity_th']} | Alignment threshold: {self['alignment_th']}",
            "(E) Sequence matching",
            f"Recall threshold: {self['recall_th']} | Min. number of hits: {self['min_hits']} | Overlap threshold: {self['overlap_th']}"
        ]


    def print_terminal(self):
        for line in self._lines():
            print(line)


    def print_info(self):
        for line in self._lines():
            logger.info(line)