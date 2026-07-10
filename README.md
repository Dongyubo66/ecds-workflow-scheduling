# ECDS: Green-Energy-Aware Workflow Scheduling

This repository is the public research artifact for **"Green-Energy-Aware Online Multi-Objective Scheduling for Dynamic DAG Workflows."** It implements Energy- and Carbon-aware Dynamic Scheduling (ECDS) and the HEFT, GREENHEFT, and MOHEFT baselines for static and dynamic DAG-workflow scheduling.

ECDS uses an event-driven simulator, a site-aware energy/carbon model, resource profiles, and four scheduling objectives: makespan, energy, carbon, and green-energy use. The artifact includes the exact configuration families, fixed seeds, result summaries, and figure scripts retained from the experiment workspace.

## Repository layout

- `src/` – simulator, workflow loader, energy/resource models, schedulers, and metrics.
- `configs/` – static, dynamic, four-objective, cross-family, and ablation configurations.
- `experiments/` – cross-family runners and aggregation utilities.
- `scripts/` – workflow download and verification utilities.
- `results/` – existing raw-study CSVs, aggregate CSVs, final-submission summaries, and preserved figures.
- `plots/` – plotting programs used for the included Figure 6–10 artifacts and cross-family plots.
- `manifests/` – checksums for the non-redistributed workflow instances and release files.

## Installation

Python 3.12 was used for the audited local smoke test. Create an isolated environment and install the pinned runtime dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# POSIX shells: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Obtain workflow instances

Workflow JSON files are not included because the archived upstream repository did not provide an explicit license file that permits redistribution. Download the exact upstream version first:

```bash
bash scripts/download_workflows.sh
# or on Windows PowerShell:
pwsh -File scripts/download_workflows.ps1
python scripts/verify_workflows.py
```

The scripts obtain `wfcommons/pegasus-instances` at commit `813a2a7d3e7273200805e89f5475f9126d903eab` (the upstream `v1.4` revision). See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) for provenance and limitations.

## Quick start

Run the existing static single-instance configuration after obtaining the data:

```bash
python src/main.py --cfg configs/static.yaml --out results/runs/static_smoke.csv
```

The dynamic counterpart is:

```bash
python src/main.py --cfg configs/dynamic.yaml --out results/runs/dynamic_smoke.csv
```

These commands preserve the checked-in result artifacts; they write only the explicitly selected output file.

## Full reproduction

The cross-family runs use ECDS, HEFT, GREENHEFT, and MOHEFT over Montage, Epigenomics, and Seismology instances. They may take substantial time; no duration claim is made because an end-to-end runtime was not recorded for this release.

```bash
python experiments/run_grid.py --plan configs/dynamic_cross_family_all4_full.yaml
python plots/build_dynamic_main_summary_all4.py \
  --input-dir results/dynamic_cross_family_all4/raw \
  --output results/final_submission_snapshot/dynamic_main_summary_all4.csv
python plots/plot_cross_family_main_all4.py \
  --input results/final_submission_snapshot/dynamic_main_summary_all4.csv \
  --out-dir results/final_submission_snapshot/figures
```

For a smaller cross-family run, use `configs/dynamic_cross_family_all4_quick.yaml`. Static, dynamic, ablation, seed, input, and expected-output mappings are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Ablations and figures

Existing ablation plans are `configs/plans/ablation_static.yaml` and `configs/plans/ablation_dynamic.yaml`. Existing plotting programs and their corresponding retained inputs are listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The preserved final-submission CSVs and Figures 6–10 are in `results/final_submission_snapshot/`; they are copied artifacts, not regenerated values.

## Data source and limitations

The implementation expects WfCommons-format JSON workflow instances. The upstream data are acquired separately and verified by checksum; this repository does not claim to have created them. The simulator’s site profiles and machine parameters are experimental model inputs, not measurements of a production cloud. Results can vary across platforms and dependency versions; this release does not alter the originally retained CSV values to force agreement.

## Citation

The source repository is [Dongyubo66/ecds-workflow-scheduling](https://github.com/Dongyubo66/ecds-workflow-scheduling). See [CITATION.cff](CITATION.cff) for citation metadata. The Zenodo version-specific DOI will be added only after the corresponding archive actually exists.

## License

The original code released here is available under the [MIT License](LICENSE). This license does not apply to third-party workflow data.
