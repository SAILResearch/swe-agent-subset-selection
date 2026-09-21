# Replication package: Trajectory-Aware Benchmark Subset Selection for Cost-Efficient Software Engineering Agent Regression Testing

The paper proposes selecting a small subset of benchmark instances for regression-testing software engineering agents:
instances are grouped by their historical pass/fail outcomes, and within each group the subset is chosen
deterministically from sanitized trajectory embeddings. It compares 76 subset selection configurations by how closely
the subset's resolve rate tracks the full benchmark's resolve rate on later runs (RMSE and worst-case error, MaxErr).

This package contains the study's result files (`results/`), the scripts that turn result files into the paper's
tables, figures and in-text numbers (one folder per research question), and the pipeline that produces result files
from raw trajectories for all three datasets (`pipeline/`).

## Three ways to use this package

Install the requirements first (tested with Python 3.13): `pip install -r requirements.txt` for the first mode,
additionally `pip install -r pipeline/requirements.txt` for the other two. Every command prints its plan and an
estimated run time before it starts; add `--dry-run` to see only that.

1. **From the shipped results** (nothing to download):

   ```
   python reproduce.py results                 # everything
   python reproduce.py results --rq 2          # one research question: 1, 2, 3, 4 or supporting
   ```

2. **Pipeline for one research question and dataset** (the dataset's published vectors are downloaded first if
   `data/<dataset>/vectors/` does not exist, see "Data"):

   ```
   python pipeline/fetch.py --dataset multi_model --vectors                                     # also run by the next command
   python reproduce.py pipeline --dataset multi_model --rq 2
   python reproduce.py pipeline --dataset multi_model --rq 2 --windows 2 --distributions 0-4    # a small part
   python reproduce.py pipeline --dataset single_setup --rq 2 --run-groups 5 --windows 2        # a small part
   ```

   `--rq` takes 1, 2, 3, 4 or supporting. Research question 4 (cost) is computed for multi_model only; the supporting
   numbers use multi_model and single_setup, and for multi_model they need the parsed trajectories (`--from raw`).

3. **Whole pipeline** (every research question available for the dataset):

   ```
   python reproduce.py pipeline --dataset multi_model --all [--configurations all] [--from raw]
   ```

Pipeline mode writes regenerated result files and outputs under `regenerated/` and never into `results/`; work already
done in that folder is reused, so `--rq 2` after `--rq 1` runs only what is missing. `--configurations reported`
(default) runs the configurations reported in the paper: the six baselines and the Stability-Stratified control, the ten
embedding-within-strata methods, the per-seed baseline draws, the pure-embedding family, and the shortlist and clustering
configurations named in Table 7; `all` runs all 76. `--from raw` adds fetch, run grouping (single-setup), parse,
sanitize and the trajectory features of the clustering configurations before the selection stage; embedding is a
separate step of many hours that runs only with `--embed`, otherwise the published vectors are used. Research question 4
runs the cost stages (the cost per trajectory is recomputed when the raw trajectories are in `data/multi_model/raw/`,
otherwise read from `results/`); the supporting numbers run the checks of the synthetic distributions. A part restricted
with `--windows` must contain the window sizes the research-question script reads (research question 1 reads W = 1
and at least one W >= 2, research question 4 reads W = 2); `reproduce.py` stops with a message otherwise.

Estimated run times (Apple M1 Pro, 10 cores, 16 GB, 10 worker processes; "measured": every part of the sum was measured
in a full run of that part on this machine; the other values are estimates scaled from measured parts). The tables per
research question and dataset, for `--configurations reported` and `all`, are in `pipeline/README.md`.

| Mode | multi_model | multi_agent | single_setup | Download first |
|---|---|---|---|---|
| results (all) | under 1 min for all datasets together (measured) | | | nothing |
| pipeline, one research question, reported configurations | 0.3 h (RQ4) to 5.7 h (RQ3) (measured) | 5.9 to 11.0 h (estimated) | 0.7 to 2.0 h (estimated) | vectors (`fetch.py --vectors`) |
| whole pipeline, reported configurations | 5.8 h (measured) | 11.0 h (estimated) | 2.0 h (estimated) | vectors (`fetch.py --vectors`) |
| whole pipeline, all 76 configurations | 27.7 h (partly estimated) | 51.5 h (estimated) | 2.0 h (estimated; 60 configurations) | vectors (`fetch.py --vectors`) |
| `--from raw` adds | about 5 min | about 5 min | about 40 min | raw trajectories (`pipeline/fetch.py`) |
| `--embed` adds | 7 to 35 h | 7 to 39 h | 53 to 300 h | raw trajectories |

## Datasets

| Name in this package | Paper name | Data |
|---|---|---|
| `single_setup` | Single-setup | 45 reruns of OpenHands with Qwen3-Coder on SWE-rebench (3,188 instances, 25,279 trajectories) |
| `multi_model` | Multi-model | 6 OpenHands runs with different models on SWE-Bench Verified (500 instances, 3,000 trajectories) |
| `multi_agent` | Multi-agent | 7 runs across five agent frameworks on SWE-Bench Verified (3,500 trajectories) |

## Layout

```
README.md, LICENSE, requirements.txt
reproduce.py                        entry point (modes: results, pipeline)
rq1_baselines/reproduce.py          + outputs/   Tables 3, 4; Figure 5; reproduced values of Section 6.1
rq2_trajectory_aware/reproduce.py   + outputs/   Tables 5, 6; Figure 6; single-setup table; sensitivity files; Section 6.2
rq3_ablation/reproduce.py           + outputs/   Table 7; configuration counts; determinism validation; Section 6.3
rq4_cost/reproduce.py               + outputs/   Tables 8, 9; Section 6.4
supporting/reproduce.py, supporting/single_setup_consistency.py + outputs/   Sections 5.1, 5.2.1, 8.3, 8.4; single-setup across run groups
common/                             shared helpers; paper_values.py holds the values printed in the paper's tables
                                    (used only for the "value in the paper" column of the outputs, in no computation)
results/                            the study's result files, by dataset
pipeline/                           raw trajectories -> result files, all three datasets (see pipeline/README.md)
```

Every research-question script accepts `--results <folder>` (default `results/`) and `--outputs <folder>` (default its
own `outputs/`). The two gzipped result files (24 MB and 34 MB; 104 MB and 123 MB of JSON) are decompressed on the fly.
In the result files, keys containing "fold" (for example `n_folds`, `fold_mean`) refer to temporal splits.

## Where each paper item is reproduced

| Paper item | Script | Output |
|---|---|---|
| Table 3, Table 4, Figure 5 | `rq1_baselines/reproduce.py` | `rq1_baselines/outputs/tables/table3_baseline_rmse.*`, `table4_baseline_maxerr.*`, `figures/figure5_baselines_multi_model.pdf` |
| Table 5, Table 6, Figure 6 | `rq2_trajectory_aware/reproduce.py` | `rq2_trajectory_aware/outputs/tables/table5_embedding_rmse.*`, `table6_embedding_maxerr.*`, `table6_paper_vs_reproduced.csv`, `figures/figure6_centroid_vs_consist_multi_model.pdf` |
| Single-setup RQ2 results; sensitivity to window size; pooled vs time-series | `rq2_trajectory_aware/reproduce.py` | `rq2_trajectory_aware/outputs/tables/single_setup_embedding.*`, `rq2_trajectory_aware/outputs/centroid_vs_strongest_baseline_by_window.csv`, `rq2_trajectory_aware/outputs/pooled_vs_time_series.csv` |
| Table 7, Figure 7 counts, Finding 3.4 | `rq3_ablation/reproduce.py` | `rq3_ablation/outputs/tables/table7_ablation.*`, `configuration_counts.csv`, `determinism_band.csv`, `stability_vs_difficulty_stratified.csv` |
| Table 8, Table 9 | `rq4_cost/reproduce.py` | `rq4_cost/outputs/tables/table8_cost_profile.*`, `table9_cost_savings.*` |
| Numbers in the text of Sections 6.1 to 6.4 | the four research-question scripts | `<folder>/outputs/reproduced_values.{md,csv}` |
| Numbers in Sections 5.1, 5.2.1, 8.3, 8.4 | `supporting/reproduce.py` | `supporting/outputs/reproduced_values.{md,csv}` |
| Single-setup results across run groups | `supporting/single_setup_consistency.py` | `supporting/outputs/single_setup_consistency/consistency_summary.{md,csv}` |

Tables 1 and 2 and Figures 1 to 4 are illustrative. Each `reproduced_values` file has one row per number stated in its
section of the paper: the location, the statement, the value in the paper, the value reproduced from the result files,
and the file and key or formula used.

## Which result files each research question reads

| Research question | Result files (per dataset folder under `results/`) |
|---|---|
| RQ1 | `aggregated_results.json`, `per_seed_maxerr_all_baselines.json`, `per_seed_maxerr_extract.npz` |
| RQ2 | `aggregated_results.json`, `per_seed_maxerr_all_baselines.json`, `per_seed_maxerr_extract.npz`, `per_seed_maxerr_vs_consistency_stratified.json` |
| RQ3 | `aggregated_results.json`, `determinism_band.json`, `per_distribution_mean_rmse.json`, `per_seed_maxerr_extract.npz` |
| RQ4 | `cost_profile.json`, `cost_per_trajectory.json`, `cost_centroid_pooled_10pct.json` |
| Supporting | `synthetic_distributions.json`, `cross_run_resolve_rates.json`, `difficulty_level_validity_per_distribution.csv`, `instance_churn_per_distribution.csv`, `per_distribution_mean_rmse.json`, `cost_profile.json`, `aggregated_results.json`, `run_groups.json` |

The two large `aggregated_results.json` files are shipped gzipped. The cost files and the files of the synthetic
distributions exist for multi-model; single-setup also has `run_group_<N>/aggregated_results.json` for the run groups
5 to 9. `pipeline/README.md` lists which pipeline stage writes each file.

`results/<dataset>/synthetic_distributions.json` are the 1,000 synthetic distributions used in the study; the selection
stage of the pipeline reads them by default.


## Data

- Vectors (needed for pipeline mode): the Hugging Face dataset repository `Mahmoud-queens/swe-agent-subset-selection-vectors`
  (https://huggingface.co/datasets/Mahmoud-queens/swe-agent-subset-selection-vectors; set as `VECTORS_REPOSITORY` in
  `pipeline/config.py`) holds `multi_model_vectors.tar.gz` (391 MB), `multi_agent_vectors.tar.gz`
  (473 MB) and `single_setup_vectors.tar.gz` (4.7 GB), an md5 manifest per archive (`<dataset>_vectors.md5`, one line per
  file in the archive) and `single_setup_repository_map.json`. `python pipeline/fetch.py --dataset <name> --vectors`
  downloads the dataset's archive, unpacks it to `data/<dataset>/`, which gives
  `data/<dataset>/vectors/{pooled,timeseries}/` and, for multi-model and multi-agent, `data/<dataset>/features/`
  (trajectory features read by some clustering configurations), checks every unpacked file against the manifest and,
  for single-setup, places the repository map as `data/single_setup/repository_map.json`. Files downloaded by hand can
  be passed with `--source FOLDER`.
- Raw trajectories (only for `--from raw`): `python pipeline/fetch.py --dataset <name>` downloads them (see
  `pipeline/README.md`, "Stage 0: fetch").
- Sources: single-setup: https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories ; multi-model and
  multi-agent: https://github.com/SWE-bench/experiments

## Citation

TODO


