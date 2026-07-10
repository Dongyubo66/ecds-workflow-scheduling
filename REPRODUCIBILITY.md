# Reproducibility Guide

## Audited environment

The local smoke test used Windows and Python 3.12.7 with the package versions pinned in `requirements.txt`. No CPU, memory, accelerator, or full-experiment wall-clock record was available in the audited workspace; this document deliberately does not invent one.

## Data preparation

Run one of the following from the repository root:

```bash
bash scripts/download_workflows.sh
# or
pwsh -File scripts/download_workflows.ps1
python scripts/verify_workflows.py
```

The verification program checks the six workflow files referenced by the release configurations against `manifests/workflow_sha256.txt`.

## Fixed seeds and configurations

- `configs/static.yaml`: static single-instance setup; `seeds: [1, 2, 3, 4, 5]`.
- `configs/dynamic.yaml`: dynamic single-instance setup; `seeds: [1, 2, 3, 4, 5]`, release horizon `T=100`.
- `configs/dynamic_all4_base.yaml`: four-objective dynamic study base; `seeds: [1, 2, 3, 4, 5]`.
- `configs/dynamic_cross_family_all4_full.yaml`: four schedulers, six cross-family instances, five seeds.
- `configs/dynamic_cross_family_all4_quick.yaml`: reduced cross-family configuration.
- `configs/plans/ablation_static.yaml` and `configs/plans/ablation_dynamic.yaml`: ECDS ablations with seeds `[1, 2, 3]`.

## Commands

Single static and dynamic runs:

```bash
python src/main.py --cfg configs/static.yaml --out results/runs/static_smoke.csv
python src/main.py --cfg configs/dynamic.yaml --out results/runs/dynamic_smoke.csv
```

Cross-family static and dynamic runs retained from the workspace:

```bash
python experiments/run_grid_families_static.py
python experiments/run_grid_families_dynamic.py
python experiments/run_grid_families_dynamic_seeds10.py
```

The generic experiment-grid runner supports the full and quick plans:

```bash
python experiments/run_grid.py --plan configs/dynamic_cross_family_all4_quick.yaml
python experiments/run_grid.py --plan configs/dynamic_cross_family_all4_full.yaml
```

Ablations:

```bash
python experiments/run_grid.py --plan configs/plans/ablation_static.yaml
python experiments/run_grid.py --plan configs/plans/ablation_dynamic.yaml
```

## Retained result artifacts

The release preserves rather than recomputes the result artifacts used by the plotting programs:

- `results/dynamic_cross_family_all4/` – existing per-run cross-family CSVs. Source-machine execution manifests were excluded because they contained local absolute paths.
- `results/study/` – existing dynamic Pareto and budget raw/summary CSVs, including the inputs to HV, IGD, and additive-epsilon summaries.
- `results/final_submission_snapshot/` – retained tables, summaries, and figures used for final submission.
- `results/summaries/` – aggregate static/dynamic cross-family summaries.

## Figure generation

The following commands use existing retained inputs. They overwrite only their named outputs under `results/final_submission_snapshot/figures`.

```bash
python plots/plot_cross_family_main_all4.py --input results/final_submission_snapshot/dynamic_main_summary_all4.csv --out-dir results/final_submission_snapshot/figures
python plots/plot_fig6_pareto_projection_v6.py
python plots/plot_budget_runtime_richer_v2.py
python plots/plot_budget_igd_eps_richer_v2.py
python plots/build_and_plot_fig9_fig10.py
python plots/plot_ablation.py
```

The artifact retains the final-submission figures labelled 6–10. Figures 3–5 were not uniquely identifiable from the audited workspace as a single final plotting pipeline; no replacement plot data or scripts have been fabricated. The source `plots/` directory contains only the existing scripts selected for the retained final artifacts.

## Expected outputs

- Single-run commands write one CSV to the path passed with `--out`.
- Grid runs write raw files and manifests to the plan’s `experiment.output_dir`.
- The figure commands write PNG/PDF/SVG files to `results/final_submission_snapshot/figures` as defined by the original scripts.

## Non-mutating smoke test

After downloading workflows, run a single existing configuration while directing the output to a temporary location. On PowerShell:

```powershell
$out = Join-Path $env:TEMP 'ecds-smoke.csv'
python src/main.py --cfg configs/static.yaml --out $out --quiet
Remove-Item -LiteralPath $out
```

This exercises the loader, simulator, energy/resource model, scheduler construction, and CSV writer without modifying checked-in data.
