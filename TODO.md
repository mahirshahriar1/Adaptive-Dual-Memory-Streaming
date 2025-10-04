## TODO

### Highest Impact Next Steps

#### Immediate (This Week)
- Perform ablation: dynamic vs static sink to demonstrate dynamic sink improvements.
- Profile throughput to identify compression bottlenecks (e.g., SVD, attention remapping).

#### Short-term (Next 2 Weeks)
- Implement attention-based importance scoring to replace the value norm heuristic.
- Tune compression interval to balance quality and speed.
- Conduct cross-architecture validation by running Pythia, Falcon, and MPT benchmarks.

#### Medium-term (Research Contribution)
- Train with distillation to learn compression policy from a full-context teacher.
- Develop learned anchoring by training positional assignments (instead of grid/mean).
- Explore adaptive budgets with per-layer or per-head dynamic allocation.
- Perform task-specific tuning for Q&A, summarization, and chat optimization.

#### Paper-Ready
- Complete comprehensive ablations for every hyperparameter, including error bars.
- Analyze failure cases: when does ADMS underperform (e.g., short contexts, specific patterns)?
- Establish memory/speed/quality Pareto frontier to guide user configuration choices.
- Prepare open-source release with clean code, documentation, and pretrained policies.