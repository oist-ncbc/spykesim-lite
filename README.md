# spykesim-lite

`spykesim-lite` is a reinterpretation of [spykesim](https://github.com/oist-ncbc/spykesim) based on the original source code and the description provided by [Watanabe et al. (2019)](https://www.frontiersin.org/articles/10.3389/fninf.2019.00039). This repository offers functions for measurement of similarity between two time series of multi-neuron spiking activity and provides the backbone for the analysis in the manuscript "Cross-layer non-random networks associated with reward-earning motor behavior" (provisional title, under preparation, citation to be added later).

Among the modifications provided by this repository, we highlight:

- normalization of the edit similarity matrix;
- calculation of the eligibility matrix for masked clustering;
- utilization of Affinity Propagation for clustering.

## Requirements

This library was tested on Python 3.12.10 (Mac) and Python 3.11.4 (Linux). The analysis reported in the manuscript used the Linux environment. The following libraries are required in the local environment (in parenthesis: versions used when running on Python 3.12.10):

- `numpy` (2.3.5)
- `scipy` (1.18.1)
- `pandas` (3.0.5)
- `scikit-learn` (1.9.0)
- `matplotlib` (3.11.1)
- `joblib` (1.5.3)
- `numba` (0.62.1)

Install with:

```sh
pip install -r requirements.txt
```

> [!NOTE]
> If `numba`/`llvmlite` fails to build on Mac due to a Homebrew LLVM version mismatch, try `pip install --only-binary=:all: numba` or remove Homebrew LLVM from `PATH` before installing.

## How to run

### Data preparation

The current code is tailored to process the cross-layer rat motor cortex dataset previously described in [Isomura et al. (2009)](https://www.nature.com/articles/nn.2431), to be published publicly as **10.5281/zenodo.22055254** (currently a private draft). Each session contains `All.clu.X` and `All.res.X` files, with X = 1, 2 indicating the index of the tetrode (1: L2/3, also referred to as superficial layer; 2: L5, or deep layer).

Set `data_root = "/path/to/data"` (or the dataset path variable) in `data_manager.py`, line 13, before running. Choose which session to analyze using the configuration file (see below).

### Configuration file

The scripts expect a configuration file in JSON format. The specific session, start time of the stage of interest, and the exponential gap penalty to be used in the edit similarity calculation (see manuscript for details) are set in this file.

> [!TIP]
> Edit similarity calculation (computation and memory) scales as Θ(T^2) with the number of segmented windows T. We suggest starting with a short duration for testing purposes.

#### Example

See `src/config_example.json`. A description of the parameters is provided below.

```json
{
    "dataset": "071102",
    "start_time_sec": 100,
    "duration_min": 1,
    "max_rate_Hz": 3,

    "window_ms": 100,
    "slide_ms": 100,
    "bin_ms": 1,

    "alpha": 0.0693,
    "match_penalty": 0,
    "min_neurons": 3,

    "rand_state": 12345,
    "damping": 0.9,
    "max_iter": 2000,
    "min_cluster_size": 10,

    "spike_relevance_th": 0.7,
    "similarity_th": 1,
    "alignment_th": 1,

    "recall_th": 1,
    "overlap_th": 2,
    "min_hits": 5
}
```

### Run analysis

After setting the configuration file, analysis can be run using the following commands:

```sh
python src/run_simmat.py --config config_example.json
```

This runs the symmetric edit similarity calculation and produces a normalized similarity matrix (float32) and an eligibility matrix (boolean).

```sh
python src/run_clustering.py --config config_example.json
```

This runs Affinity Propagation on the similarity matrix using the eligibility matrix as a mask to floor the similarity of segmented window pairs with low spike count. It produces a dictionary with information regarding detected clusters.

```sh
python src/run_profiles.py --config config_example.json
```

This runs profile extraction (i.e., pattern detection with temporal flexibility) using clustering results and produces a dictionary with profile information and detected sequences.

All outputs are written to a local folder following the format set by the configuration file:

```sh
./output/dataset{dataset}_start{start_time_sec}_dur{duration_min}_fr{max_rate_Hz}_win{window_ms}_slide{slide_ms}_bin{bin_ms}_alpha{alpha}_matchpen{match_penalty}_minneu{min_neurons}
```

For the rest of the analysis introduced in the manuscript, uploaded on a different repository (under preparation), we use the `final_prof_dict_{suffix}.npz` file written by `run_profiles.py`.

## Parameter description

The following parameters are expected when running the pipeline. Examples provided refer to the Isomura et al. (2009) dataset.

- `dataset`: ID of the dataset (or session) of interest. Ex: `071102`
- `start_time_sec`: start time of the interval of analysis in seconds. Ex: `100`
- `duration_min`: duration of the interval of analysis in minutes. Ex: `20`
- `max_rate_Hz`: maximum average firing rate allowed in Hertz. Neurons above this threshold will be removed from the analysis. Ex: `3`
- `window_ms`: length of the time window of interest in milliseconds. Patterns to be found will be restricted to this window length. Ex: `100`
- `slide_ms`: length of the sliding window in milliseconds. Ex: `100`
- `bin_ms`: length of the binning window in milliseconds. Ex: `1`
- `alpha`: exponential penalty factor for edit similarity calculation. It is linked to the average jitter allowed (measured in bin counts) through the equation below. Ex: `0.0693`

$$
\alpha = \frac{\ln 2}{\text{jitter}}
$$

- **(LEGACY)** `match_penalty`: previously used to penalize bins with no matches. Ex: `0`
- `min_neurons`: minimum number of active neurons needed to define a pattern. Also used to mask the edit similarity matrix for clustering. Ex: `3`
- `rand_state`: seed used for clustering. Different preference values use incremental seeds from this root value. Ex: `12345`
- `damping`: damping factor for Affinity Propagation. Must be in the range `[0.5,1)`. Ex: `0.9`
- `max_iter`: maximum number of iterations allowed when running Affinity Propagation. Ex: `2000`
- `min_cluster_size`: minimum number of members for a cluster to be considered for further analysis. Ex: `10`
- `spike_relevance_th`: threshold for how relevant a spike must be in a cluster to be considered part of a putative pattern. Calculated as the fraction of how many cluster members the spike is present in. Ex: `0.7`
- `similarity_th`: threshold for how similar two putative patterns must be to be considered duplicates. Ex: `1`
- `alignment_th`: threshold for how well-aligned two putative patterns must be to be considered duplicates. Calculated as the fraction of total spikes present in the alignment. Ex: `1`
- `recall_th`: threshold for how many spikes of a putative pattern must be recalled by a window for it to be considered a sequence. Calculated as the fraction of total spikes present in the alignment. Ex: `1`
- `overlap_th`: threshold for how many common spikes two putative patterns must have to trigger the sequence overlap check. If triggered, this check will pick the putative pattern with the highest similarity to the window. Ex: `2`
- `min_hits`: minimum number of sequences a putative pattern must have to be saved as a profile. Ex: `5`

## License

This project is licensed under the [MIT License](LICENSE).