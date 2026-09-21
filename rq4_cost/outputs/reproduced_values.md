# Section 6.4 (RQ4): values stated in the paper and reproduced from `results/`

One row per number stated in the paper; values are recomputed from `results/`.

| Paper location | Statement | Value in the paper | Reproduced value | How it is computed |
|---|---|---|---|---|
| Table 8 | All printed cells | 35 cells | 33 of 35 cells equal the printed value; 2 of the 2 differing cells differ only in the last printed digit; largest absolute difference 1.00; differing: Sonnet 4 (May '25), Total (M): printed '945', reproduced '944'; Aggregate, Total (M): printed '3,442', reproduced '3,441' | generated table compared with common/paper_values.py |
| Table 9 | All printed cells | 16 cells | 16 of 16 cells equal the printed value | generated table compared with common/paper_values.py |
| Section 6.4 text | Trajectories measured | 3000 | 3000.0000 | aggregate.num_trajectories |
| Section 6.4 text | Total tokens across all six runs (billions) | 3.44 | 3.4409 | aggregate.total_triangular_tokens |
| Section 6.4 text | Cheapest full run (M tokens) | 335 | 335.3444 | agent_profiles.*.total_triangular_tokens |
| Section 6.4 text | Most expensive full run (M tokens) | 945 | 944.4681 | agent_profiles.*.total_triangular_tokens |
| Section 6.4 text | Input share of the token volume (%) | 99.3 | 99.3000 | aggregate.input_share_pct |
| Section 6.4 text | Sonnet 4 steps per instance | 69.7 | 69.7000 | mean_steps |
| Section 6.4 text | GPT-5 steps per instance | 31.2 | 31.2000 | mean_steps |
| Section 6.4 text | Step ratio Sonnet 4 / GPT-5 | 2.2 | 2.2340 | ratio of mean_steps |
| Section 6.4 text | Sonnet 4 mean cost per instance (M) | 1.89 | 1.8889 | per_instance.mean |
| Section 6.4 text | GPT-5 mean cost per instance (M) | 0.70 | 0.7012 | per_instance.mean |
| Section 6.4 text | Cost ratio Sonnet 4 / GPT-5 | 2.7 | 2.6940 | ratio of per_instance.mean |
| Approach | Tool outputs truncated above (tokens) | 10000 | 10000.0000 | metadata.max_observation_tokens |
| Approach | Random subsets drawn per run and subset size | 10000 | 10000.0000 | metadata.n_monte_carlo_simulations |
| Finding 4.1 | Per-instance cost differs by up to (x) | 492 | 492.7000 | aggregate.per_trajectory.cost_ratio_max_to_min |
| Finding 4.1 | A k% subset consumes k% of the token cost on average (to one decimal place) | k% | 5%: 5.0, 10%: 10.02, 20%: 20.01, 30%: 29.99 | savings_empirical.aggregate.*.weighted_mean_cost_share_pct |
| Finding 4.2 | Centroid Pooled 10% subset consumes (% of the run's token cost) | 10.48 | 10.4800 | 100 - summary.overall.aggregate_saved_pct |
| Finding 4.2 | ... which sits inside the middle 95% of the Random draws at the same size | inside | 10.48% vs [5.72%, 15.04%] | Table 9 range |
| Finding 4.2 | A 10% subset cuts token consumption to roughly (M) | 345 | 344.6223 | savings_empirical.aggregate.10pct.estimated_subset_tokens |
