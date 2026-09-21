# Pipeline: from raw trajectories to the result files

Stages, in order: fetch -> (group runs) -> parse -> sanitize -> embed -> select -> evaluate, plus the cost stages and
the checks of the synthetic distributions. Every stage script takes `--dataset` and explicit input and output folders
(defaults under `data/<dataset>/<stage>/`, which is not tracked), prints what it read and wrote, and stops with a
message when an input is missing.

Install the pipeline requirements with `pip install -r pipeline/requirements.txt` (tested with Python 3.13).

Default data layout (all under the package root):

```
data/<dataset>/raw/             raw trajectories (fetch.py)
data/single_setup/grouped/      run groups (group_runs.py)
data/<dataset>/parsed/          unified schema (parse.py)
data/<dataset>/sanitized/       masked trajectories (sanitize.py)
data/<dataset>/vectors/         pooled/ and timeseries/ (fetch.py --vectors, or embed.py)
data/<dataset>/features/        trajectory features of the clustering configurations (fetch.py --vectors, or extract_features.py)
data/single_setup/repository_map.json      trajectory id -> instance, repository (fetch.py --vectors, or build_repository_map.py)
data/<dataset>/distributions/   synthetic distributions (generate_distributions.py)
data/<dataset>/selection/       selection results
data/<dataset>/results/         result files
```

## How `reproduce.py pipeline` chains the stages

`python reproduce.py pipeline --dataset <name> --rq <N>` (package root) runs, for the families that the research
question needs, the selection stage (`run_selection.py` for multi-model and multi-agent; the `select_*.py` scripts and
`determinism.py` for single-setup), then the evaluation scripts, the cost stages or the distribution checks, and finally
the research-question script on the regenerated result files. It starts with `fetch.py --vectors`, which downloads
the published vectors when `data/<dataset>/vectors/` does not exist. With `--from raw` it first runs `fetch.py`,
`group_runs.py` (single-setup), `parse.py --validate`, `sanitize.py`, and `build_repository_map.py` (single-setup) or
`extract_features.py` (the other two); `embed.py` runs only with `--embed`. It passes the study's distributions file
(`results/<dataset>/synthetic_distributions.json`) to every stage that needs one. Every stage can also be run alone.

| Research question | Selection families | Per-seed draws | Evaluation scripts | Further stages | Datasets |
|---|---|---|---|---|---|
| RQ1 | baselines, embedding within strata (only `centroid_pooled`) | yes | yes | none | multi_agent, multi_model, single_setup |
| RQ2 | baselines, embedding within strata | yes | yes | none | multi_agent, multi_model, single_setup |
| RQ3 | baselines, embedding within strata, pure embedding, shortlist, clustering | yes | yes | none | multi_agent, multi_model, single_setup |
| RQ4 | embedding within strata (only `centroid_pooled`) | no | no | `cost_per_trajectory.py`, `cost_profile.py`, `cost_subsets.py` | multi_model |
| Supporting | baselines, embedding within strata | no | yes | `generate_distributions.py`, `distribution_validity.py` | multi_model, single_setup |

RQ1 runs Centroid Pooled because the per-seed result files report each baseline next to it. Single-setup has no
clustering family. The evaluation scripts are `evaluate.py`, `evaluate_per_seed.py` (when the per-seed draws are run)
and `evaluate_distributions.py` (multi-model, multi-agent) or `summarize_run_groups.py` (single-setup). RQ4 recomputes
the cost per trajectory when `data/multi_model/raw/` exists and otherwise reads it from `results/`. The supporting
numbers of multi-model need the parsed trajectories (`--from raw`, or an existing `data/multi_model/parsed/`).

Result files written by the stages that follow the selection stage:

| Stage | Result files |
|---|---|
| `evaluate.py` | `aggregated_results.json` |
| `evaluate_per_seed.py` | `per_seed_maxerr_all_baselines.json`, `per_seed_maxerr_vs_consistency_stratified.json`, `per_seed_maxerr_extract.npz`, `determinism_band.json`, `determinism_band_maxerr.json` |
| `evaluate_distributions.py` | `per_distribution_mean_rmse.json` |
| `summarize_run_groups.py` | `run_groups.json` |
| `cost_per_trajectory.py` | `cost_per_trajectory.json` |
| `cost_profile.py` | `cost_profile.json` |
| `cost_subsets.py` | `cost_centroid_pooled_10pct.json` |
| `distribution_validity.py` | `cross_run_resolve_rates.json`, `cross_run_resolve_rates.csv`, `difficulty_level_validity_per_distribution.csv`, `instance_churn_per_distribution.csv` |

For single-setup, `evaluate.py` also writes `run_group_<N>/aggregated_results.json` and `evaluate_per_seed.py` writes
the two per-seed files only. `evaluate.py` additionally writes `evaluation_manifest.json` (families and configurations
that were evaluated). `results/<dataset>/synthetic_distributions.json` is an input of the study (see "Synthetic
distributions").

Run time of `reproduce.py pipeline` from the published vectors, per research question (hours on an Apple M1 Pro,
10 cores, 16 GB, 10 worker processes; "measured": every part of the sum was measured in a full run of that part on this
machine; "estimated": scaled from measured parts, see "Run times of the stages"; `--from raw` adds about 5 minutes,
40 minutes for single-setup):

multi_model

| Research question | `--configurations reported` | `--configurations all` |
|---|---|---|
| RQ1 | 2.1 h (measured) | 2.1 h (measured) |
| RQ2 | 3.0 h (measured) | 3.0 h (measured) |
| RQ3 | 5.7 h (measured) | 27.7 h (partly estimated) |
| RQ4 | 0.3 h (measured) | 0.3 h (measured) |
| Supporting | 2.3 h (measured) | 2.3 h (measured) |
| all of the above (`--all`) | 5.8 h (measured) | 27.7 h (partly estimated) |

multi_agent

| Research question | `--configurations reported` | `--configurations all` |
|---|---|---|
| RQ1 | 5.9 h (estimated) | 5.9 h (estimated) |
| RQ2 | 5.9 h (estimated) | 5.9 h (estimated) |
| RQ3 | 11.0 h (estimated) | 51.5 h (estimated) |
| all of the above (`--all`) | 11.0 h (estimated) | 51.5 h (estimated) |

single_setup

| Research question | `--configurations reported` | `--configurations all` |
|---|---|---|
| RQ1 | 1.0 h (estimated) | 1.0 h (estimated) |
| RQ2 | 1.0 h (estimated) | 1.0 h (estimated) |
| RQ3 | 2.0 h (estimated) | 2.0 h (estimated) |
| Supporting | 0.7 h (estimated) | 0.7 h (estimated) |
| all of the above (`--all`) | 2.0 h (estimated) | 2.0 h (estimated) |

## Datasets

`config.py` defines each dataset: its trajectory format (parser), its sanitization rules and its runs in chronological
order.

`multi_model`: six OpenHands runs on SWE-Bench Verified. `multi_agent`: seven runs of five agent frameworks on
SWE-Bench Verified:

```
20241029_OpenHands-CodeAct-2.1-sonnet-20241022      OpenHands   (source run of the synthetic distributions)
20250519_trae                                       Trae
20250603_Refact_Agent_claude-4-sonnet               Refact
20250611_moatless_claude-4-sonnet-20250514          Moatless
20250612_trae                                       Trae
20250716_openhands_kimi_k2                          OpenHands
20250720_Lingxi-v1.5_claude-4-sonnet-20250514       Lingxi
```

Every stage command below takes `--dataset multi_agent` in the same way. `config.py` names the parser of each run
(`parsers/openhands.py`, `trae.py`, `refact.py`, `moatless.py`, `lingxi.py`). The source run is the same run as in
the multi-model dataset; it is sanitized with the multi-model rules, so both datasets use the same sanitized
trajectories and vectors of that run. The other six runs use `sanitization_rules/multi_agent.py`.

Instance pool of multi-agent: the repository holds 500 trajectories for five of the runs, 501 for the Moatless run
and 499 for `20250612_trae`; every file that exists is parsed. 499 instances have a trajectory in all seven runs. One
of them (`scikit-learn__scikit-learn-26194` in `20250519_trae`) has no steps and therefore no embedding. The selection
stage uses the instances that have a vector in every run: 498. Synthetic distributions keep their members from this
pool, so some populations have slightly fewer than 250 instances.

`single_setup`: reruns of OpenHands with one model on SWE-rebench, from the Hugging Face dataset
`nebius/SWE-rebench-openhands-trajectories` (revision pinned in `config.py`). The file holds several trajectories
(reruns) per instance; instances with exactly N trajectories form the run group `<N>_runs`, and run i of a group is the
i-th trajectory of each of its instances in file order (no randomness; groups with 50 instances or fewer are dropped).
The study uses the run groups with 5 to 10 reruns: 124, 321, 966, 602, 595 and 580 instances, 25,279 trajectories.
There are no synthetic distributions: the population of a run group is all of its instances, and the evaluation units
are the temporal splits of each window size W = 1..N-1 (`protocols/run_groups.py`). Runs are ordered by folder name.
Seeds are derived from window size, split and draw (see the module), the draws of a stochastic method are evaluated as
float32 arrays, and there is no clustering family and no Stability-Stratified control. The trajectories have their own
parser (`parsers/rebench_openhands.py`) and rules (`sanitization_rules/single_setup.py`: words that contain a
repository name are not masked; the parts of the repository names of all run groups of the file, listed in
`repository_names.json` by `group_runs.py`, are removed as tokens). Vector files are named by trajectory id;
`build_repository_map.py` maps them to instances and repositories for the selection stage.

## Stage 0: fetch

```
python pipeline/fetch.py --dataset multi_model --vectors [--source FOLDER]    # published vectors -> data/multi_model/
python pipeline/fetch.py --dataset single_setup                    # trajectories.parquet, about 2 GB
python pipeline/fetch.py --dataset multi_model [--runs RUN,RUN]    # six runs, about 0.8 GB
python pipeline/fetch.py --dataset multi_agent
```

Published vectors. `--vectors` downloads `<dataset>_vectors.tar.gz` and `<dataset>_vectors.md5` from the Hugging Face
dataset repository named by `VECTORS_REPOSITORY` in `config.py` (`Mahmoud-queens/swe-agent-subset-selection-vectors`,
https://huggingface.co/datasets/Mahmoud-queens/swe-agent-subset-selection-vectors), unpacks the archive
to `data/<dataset>/` (`vectors/pooled/`, `vectors/timeseries/`, and `features/` for multi-model and multi-agent),
compares the md5 of every unpacked file with the manifest, and for single-setup places
`single_setup_repository_map.json` as `data/single_setup/repository_map.json`. With `--source FOLDER` the same files are
taken from a local folder. A dataset folder that already holds `vectors/` is left untouched.

Raw trajectories. Single-setup is downloaded from Hugging Face at the pinned revision. The runs of the other two datasets are submissions
to the SWE-bench experiments repository (https://github.com/SWE-bench/experiments), which documents
`python -m analysis.download_logs evaluation/<split>/<submission>` for downloading a submission's logs and
trajectories. `fetch.py` clones that repository without file contents into `data/swe_bench_experiments/` if it is
absent, checks out `analysis/` and the dataset's submissions under `evaluation/verified/`, runs the download tool with
`--only_trajs` for every run, copies `results/` and `trajs/` of each run to `data/<dataset>/raw/<run>/`, and compares
the number of trajectories of every run with the number used in the study (500; 501 for the Moatless run, 499 for
`20250612_trae`). Runs already complete in the raw folder are skipped. Requirements: `git`, and the Python packages
`boto3` and `pyyaml`, which the download tool imports; it reads the public storage bucket without an account. With
`--use-cli` the download tool calls the AWS command-line tool (`aws s3 cp`) instead, which must then be installed and
configured with an account (`aws configure`); `fetch.py` says so when the tool is missing. Checked with the
repository at commit `40f164d`: the downloaded files of a run were identical to the ones used in the study.

## Stage 1: group runs (single-setup) and parse (Section 4.1)

```
python pipeline/group_runs.py --dataset single_setup [--run-groups 10] [--counts-only]    # data/single_setup/grouped/<N>_runs/run_<i>/
python pipeline/parse.py      --dataset multi_model --validate
python pipeline/parse.py      --dataset single_setup [--run-groups 10] --validate        # reads data/single_setup/grouped
```

`parse.py` converts the raw trajectories into the unified schema, one `unified_<instance>.json` per trajectory and a
`_parser_log.txt` per run. `--validate` checks every parsed step against the raw trajectory (tool name, action
payload and observation verbatim; recorded error consistent with the exit code) and writes
`parsing_validation.json`. Multi-model: 3,000 trajectories (134,002 steps) in 10 s; validation found 0 steps with
problems.

## Stage 2: sanitize (Section 4.2)

```
python pipeline/sanitize_discovery.py   --dataset multi_model     # Phase 1, reports for review   (44 s)
python pipeline/sanitize_leakage.py     --dataset multi_model     # Phase 2, reports for review   (27 s)
python pipeline/sanitize.py             --dataset multi_model     # Phase 3, masking              (77 s)
python pipeline/sanitize_validate.py    --dataset multi_model     # Phase 4, validation           (36 s)
python pipeline/build_repository_map.py --dataset single_setup    # data/single_setup/repository_map.json
```

Phases 1 and 2 produce reports (token frequencies, token shapes, vocabulary statistics, log-likelihood keyness of
every token between passing and failing trajectories). They were reviewed once per dataset, and the resulting stop
words, placeholder patterns and leakage terms are set in `sanitization_rules/<dataset>.py`, as described in the
paper. The masking stage reads only that rules module, never the reports, so re-running Phases 1 and 2 does not
change the sanitized trajectories. Phase 4 trains the outcome probe on the parsed and on the sanitized corpus and
scans a sample of sanitized trajectories for removed terms; its file sampling is seeded (`--seed`, default 42).

Sanitization validation of single-setup. `sanitize_validate.py` samples 5,000 files (seeded) from the sorted list of
all files below the parsed folder and, separately, below the sanitized folder, and splits them 70/30 at random by
trajectory. In the study these folders held all run groups of the file (4 to 20 reruns; written with
`group_runs.py --run-groups 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20`).

## Stage 3: embed (Section 4.3)

```
python pipeline/embed.py             --dataset multi_model [--files FILE ...]
python pipeline/verify_embeddings.py --dataset multi_model --per-run 5
python pipeline/pool_vectors.py      --input TIMESERIES_DIR --output POOLED_DIR
python pipeline/extract_features.py  --dataset multi_model
```

`embed.py` embeds every step with Nomic Embed v1.5 (pinned revision, 768 dimensions) and writes a time-series
matrix (steps x 768) and a pooled vector (first, middle and last step, mean, standard deviation: 3,840 values) per
trajectory, both float32.

The vectors used in the paper are published (see "Stage 0: fetch"), and the later stages run on those published
vectors. Embedding is slow and hardware-dependent (about 12 s per trajectory at the median on an
Apple-silicon GPU, with single trajectories taking several minutes; roughly 10 to 35 hours for multi-model), so the
datasets are not meant to be re-embedded. `verify_embeddings.py` checks the embedding script against the published
vectors on a sample: it embeds a few trajectories per run and reports, per trajectory, the maximum absolute
difference and cosine similarity of the pooled vector and of every step of the time-series matrix, without applying
a threshold. On an Apple M1 Pro, a sample of 30 multi-model trajectories (1,233 steps) gave a maximum absolute
difference of 0.0; on other hardware small floating-point differences are expected.

Vector file names carry the outcome of the run: `<OUTCOME>_<repository>_<trajectory id>_pooled.npy` and `..._ts.npy`,
with OUTCOME = `SUCCESS` or `FAIL`. The selection and evaluation stages read each run's pass/fail outcomes from this
prefix, so the files must not be renamed.

`pool_vectors.py` rebuilds pooled vectors from time-series matrices (the pooled vector is a function of the matrix
alone); it takes folders and no `--dataset`. The published pooled vectors of the 7-run group of single-setup were built
this way.

`extract_features.py` writes the ten trajectory features (steps, edit sizes, errors, files, test steps, tool shares,
observation length) read by the clustering configurations that use features; the clustering configuration reported in
the paper uses pooled embeddings and does not need them. The archives hold the feature files of the study. For
multi-model the script gives the same file. The multi-agent feature file of the study holds six of the seven runs, not
the source run, and the clustering stage averages the features over the training runs that the file holds; the script
writes all seven runs, so feature-based clustering configurations of multi-agent computed from it differ from the
shipped results.

## Synthetic distributions (Sections 5.2.1, 8.3)

```
python pipeline/generate_distributions.py --dataset multi_model    # data/multi_model/distributions/generated_distributions.json
python pipeline/distribution_validity.py  --dataset multi_model --distributions-file results/multi_model/synthetic_distributions.json
```

`generate_distributions.py` draws the 1,000 distributions from the parsed source run with a fixed seed. The draws index
into the lists of resolved and unresolved instances, so the members depend on the order of these lists; the script
lists files in sorted order, which gives the same file on every machine. The distributions files of the study
(`results/<dataset>/synthetic_distributions.json`) rest on the directory listing order of the machine that generated
them: they have the same levels, sizes, resolved counts and pairwise overlaps as the script's output, with other
instance ids. The study's files remain the input of the selection stage, so that the shipped results can be
reproduced; the selection scripts read `data/<dataset>/distributions/synthetic_distributions.json` by default
(`--distributions-file`), and `reproduce.py` passes the file in `results/`.

`distribution_validity.py` writes the resolve rate of every distribution on every run, the same per distribution with
the source run first, and the instance churn between every pair of runs.

## Stage 4: select (Sections 4.4 to 4.6, 5.2)

```
python pipeline/select_baselines.py --dataset multi_model      # 6 baselines and the Stability-Stratified control
python pipeline/select_embedding.py --dataset multi_model      # embedding-within-strata family
python pipeline/select_ablations.py --dataset multi_model      # pure-embedding, clustering and shortlist families
python pipeline/determinism.py      --dataset multi_model      # per-seed draws of the random baselines (Finding 3.4)
```

For single-setup the same scripts take `--run-groups 10` (default: the run groups 5 to 10); `select_ablations.py` then
runs pure embedding and shortlist.

Common options: `--windows 1,2,3`, `--subset-sizes 5,10,20,30`, `--seeds 500`, `--jobs N`, `--input` (vector
folder), `--distributions-file FILE`, `--distributions 0-99` (positions of the distributions to evaluate), `--output`,
`--methods centroid_pooled,fl_pooled` (evaluate only these configurations; a method's random draws do not depend on
which other methods are evaluated). `select_ablations.py` also takes `--families` and the feature files (`--features`,
`--features-meta`) used by the clustering family. Each run writes `<family>_w<W>.json` (with `_d<first>-<last>`
appended when only some distributions are evaluated) and a `.runlog.json` with the options, wall-clock time and peak
memory.

Layout of the code: `selection/` holds the protocol-independent core (data loading, instance embeddings, outcome
groups, selection algorithms, error metrics, clustering, the method families); `protocols/` holds the evaluation
protocols. `protocols/synthetic_distributions.py` (multi-model, multi-agent) evaluates every method on each of the
1,000 synthetic distributions, for every window size W and every temporal split (any W runs as training runs, all
runs after the latest training run as test runs). `protocols/run_groups.py` (single-setup) evaluates every method on
the temporal splits of each run group. All random draws are seeded from the position of the distribution (or the
window size), the split and the draw, so a partial run (some windows, some distributions) gives the same values as the
same part of a full run, and results do not depend on `--jobs`.

Instance embeddings. Pooled: the mean of an instance's pooled vectors over the training runs. Time-series: every
trajectory matrix (T steps x 768, T differs between trajectories) is brought to L = 8 steps by linear
interpolation along the step axis (a single-step trajectory is repeated), flattened to 6,144 values, and averaged
over the training runs. An instance is used only if it has a trajectory in every run of the dataset.

Deterministic methods and tie-breaking: Facility Location starts from an empty subset, where every candidate has
the same (infinite) gain, and Medoid breaks equal costs by candidate order. Both take the first candidate in the
iteration order of a Python set of integer positions, which is fixed for a given Python version (tested: CPython 3.13).

### Resumable runs

```
python pipeline/run_selection.py --dataset multi_model --distributions-file FILE --output RESULT_DIR \
       [--families baselines,embedding_strata,determinism,pure_embedding,shortlist,clustering] \
       [--methods shortlist:centroid_pooled_strata_15x,clustering:hdbscan_pooled_strata] [--windows 1,2] [--jobs N]
```

`run_selection.py` (multi-model, multi-agent) runs the selection stage in units of (family, window, block of 100
distributions), each through the entry points above in its own process. A finished unit leaves a completion record
under `RESULT_DIR/blocks/`; repeating the command skips finished units, so an interrupted run continues where it
stopped. Complete families are joined into `<family>_w<W>.json`, the determinism comparisons are computed at the end
from the joined files, and `manifest.json` lists every unit (md5 of its output, wall-clock time, peak memory, md5 of the
distributions file, code version = md5 over the source files of the selection stage). `determinism` in `--families`
stands for the per-seed baseline draws. A family named in `--methods` evaluates only those configurations. Blocks give
the same values as a single run, because every random draw is seeded by distribution position, split and draw number.

## Stage 5: evaluate (Sections 5.3, 5.4)

```
python pipeline/evaluate.py               --dataset multi_model [--input SELECTION_DIR] [--output RESULT_DIR] [--n-top 30]
python pipeline/evaluate_per_seed.py      --dataset multi_model [--input SELECTION_DIR] [--output RESULT_DIR]
python pipeline/evaluate_distributions.py --dataset multi_model [--input SELECTION_DIR] [--output RESULT_DIR]
python pipeline/summarize_run_groups.py   --dataset single_setup                      # run_groups.json
```

These turn the selection results into the result files read by the research-question scripts, with the same names,
structure and keys as the files in `results/`. The evaluation draws no random numbers.

Statistics across methods. Each ranking block of `aggregated_results.json` compares the three reference baselines
and the `--n-top` methods with the lowest mean RMSE (30 in the study); its Holm corrections and dominance ranks run
over exactly these methods, which the block lists under `methods_covered`. When only some configurations were
selected, this set, and with it the Holm-adjusted p-values, the dominance ranks and the best method per
(window, level, subset size) cell, differ from an evaluation of all 76 configurations; raw Wilcoxon p-values, effect
sizes and every per-method statistic do not. `evaluation_manifest.json` lists per family which configurations were
evaluated and whether the family is complete; the best configuration of a family is defined only for complete
families.

Evaluation of single-setup. `evaluate.py` writes one file per run group with selection results: the 10-run group (the
one reported in the paper) as `aggregated_results.json`, the others as `run_group_<N>/aggregated_results.json`. The
observations of a window-size block are its temporal splits, those of the `overall` block the window sizes; the ranking
blocks compare the three reference baselines and the 5 methods with the lowest mean RMSE (`--n-top`), listed under
`methods_covered`; methods with equal values are ordered by family name and method order. `determinism.py` writes the
per-seed draws (`determinism_<N>runs_w<W>.npz`, rows = temporal splits) and their comparison with the other methods
(`determinism_<N>runs.json`), which `evaluate.py` reads, so it runs after the selection scripts. `evaluate_per_seed.py`
writes `per_seed_maxerr_all_baselines.json` and `per_seed_maxerr_vs_consistency_stratified.json`; their `pooled` block
averages over all (run group, window size) cells of the six run groups, and the keys `fold_mean` / `fold_max` refer to
the mean and the maximum over temporal splits.

## Cost (Section 6.4, multi-model)

```
python pipeline/cost_per_trajectory.py --dataset multi_model      # reads data/multi_model/raw
python pipeline/cost_profile.py        --dataset multi_model
python pipeline/cost_subsets.py        --dataset multi_model --distributions-file results/multi_model/synthetic_distributions.json
```

`cost_per_trajectory.py` counts tokens in the raw message logs (characters / 4): the flat sum of all messages, and the
triangular cost, in which every assistant turn pays for the whole conversation before it (tool outputs capped at
10,000 tokens) and for its own message. `cost_profile.py` writes the cost profile of every run and the cost share of
10,000 seeded random subsets per run and subset size (Tables 8 and 9). `cost_subsets.py` reads the selection results of
Centroid Pooled (`embedding_strata_w<W>.json`) and the distributions file the selection was run on, and compares, for
every distribution, temporal split and test run, the cost of the 10% subset with the cost of the whole distribution;
`summary` describes W = 2, `per_window` and `all_windows` every window size. When the selection folder holds the
`manifest.json` of `run_selection.py`, the script stops if the distributions file is not the one recorded there.
On the selection results of this pipeline the subsets cost 9.78% of the full evaluation at W = 2 and 9.95% over all
window sizes; the shipped file, as computed in the study, gives 10.48%.

## Run times of the stages

Machine: Apple M1 Pro (10 cores, 16 GB), Python 3.13, `--jobs 10`. "measured" = full stage measured (select rows of
multi-model: a full run of all windows and 1,000 distributions); "slice" = measured on 20 distributions at W = 2 (all
subset sizes, 500 seeds) and scaled by the number of distributions and temporal splits (x 155 for multi-model: 31 splits
over all window sizes against 10 at W = 2; x 210 for multi-agent: 63 splits against 15; slice times of multi-agent at
W = 2: baselines 40 s, embedding within strata 33 s, pure embedding 61 s, clustering 563 s, shortlist 153 s, determinism
draws 29 s); "scaled" = scaled from multi-model by trajectory, step or character counts (parse to embed) or by the
ratio of the run times recorded in the study's own result files (select); "recorded" = run time recorded in the study's
result files (machine not recorded).

| Stage | multi_model | multi_agent | single_setup |
|---|---|---|---|
| fetch | one run (500 trajectories, 89 MB): 77 s (measured) | not measured | not measured (about 2 GB) |
| group runs | not applicable | not applicable | 10-run group: 44 s, peak memory about 7 GB (measured) |
| parse (+ validation) | 10 s (measured) | 15 s (measured; validation not timed) | 10-run group: 21 s, with validation 47 s (measured); groups 5 to 10: about 3.5 min (scaled) |
| sanitize, Phases 1 to 4 | 44 + 27 + 77 + 36 s (measured) | 41 + 29 + 71 + 66 s (measured) | 10-run group: 122 + 78 + 161 + 84 s (measured); groups 5 to 10: about 30 min (scaled) |
| embed (estimates from samples; the published vectors are the ones used in the study) | 7 to 35 h; central 12 h, of which about 1.3 h on the CPU path (sample of 30 trajectories, scaled) | 7 to 39 h; central 12 h (scaled) | 53 to 300 h; central 97 h (scaled) |
| extract features, generate distributions, distribution validity | 3 s each (measured) | 3 s each (measured) | not applicable |
| select: baselines | 1.1 h (measured) | about 2.3 h (slice estimate) | 10-run group: 3.2 min (measured); 0.1 h for groups 5 to 10 (recorded) |
| select: embedding within strata | 1.1 h (measured); Centroid Pooled alone 0.2 h (measured) | about 1.9 h (slice estimate) | 10-run group: 5.6 min (measured); 5.0 h for groups 5 to 10 (recorded) |
| select: pure embedding | 1.9 h (measured) | about 3.5 h (slice estimate) | 5.7 h (recorded) |
| select: clustering | 18 h (slice estimate); reported configuration 0.5 h (measured) | about 33 h (slice estimate) | not part of single-setup |
| select: shortlist | 4.7 h (slice estimate); reported configuration 0.3 h (measured) | about 9 h (slice estimate) | 0.5 h (recorded) |
| determinism draws | 0.8 h (measured) | about 1.7 h (slice estimate) | not measured |
| evaluate (all three scripts) | about 1 min (measured) | about 1 min (measured) | 25 s for the six run groups (measured) |
| cost (three stages) | 12 s (measured) | not applicable | not applicable |
| research-question scripts | under 1 min for all datasets together (measured) | | |

Embedding work: multi-model 134,002 steps (207 M characters; 536 steps above the context length, encoded on the
CPU), multi-agent 155,475 steps (205 M characters; 462), single-setup 1,635,294 steps (1,789 M characters; 1,018).
The embed range reflects the spread between trajectories in the sample (about 190 s per million characters on the GPU,
quartiles 100 to 225; about 9 s per CPU-path step) and single trajectories that stall for minutes on the GPU (1 of
30 in the sample). The "recorded" single-setup times should be read with a factor of about 0.9 to 1.8 for this
machine (the ratio observed for multi-model). Peak memory of the select stage: about 0.4 GB per worker (slice);
result files of a full multi-model select run: about 3 GB. Clustering dominates because K-means with silhouette
selection runs on the 6,144-dimensional time-series embeddings.
