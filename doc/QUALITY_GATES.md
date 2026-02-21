# Quality Gates

Offline quality checks are enforced by:

```bash
uv run python scripts/evals/run_all_quality_checks.py
```

Current hard thresholds:

- `p_at_5 >= 0.80`
- `coverage >= 0.95`
- `cluster_purity >= 0.75`

The command writes report JSON to `data/eval/quality_report.json`.
