# Public Release Audit

Audit date: 2026-07-10

## Source inventory

The source experiment workspace was a Git repository with a configured remote and a dirty working tree. It contained implementations of ECDS, HEFT, GREENHEFT, MOHEFT, an event-driven simulator, energy and resource models, WfCommons workflow loading, and metric/plotting utilities. It also contained static, dynamic, four-objective, cross-family, and ablation YAML configurations.

The source workspace did not contain a Python dependency lock file or environment file. Runtime imports and the audited environment were used to create `requirements.txt`; the omitted `deap` dependency was discovered during smoke testing and then included.

## Public selection

This release contains only the selected implementation files, release configurations, existing result artifacts, plotting utilities, and English public documentation. It excludes the source workspace’s IDE state, temporary configuration directories, caches, unrelated logs, large workflow JSON files, and source-control metadata.

One cross-family execution manifest was also excluded because it recorded source-machine absolute paths. The underlying per-run result CSVs were retained unchanged.

Absolute local workflow paths in public configuration files were changed to `data/pegasus-instances`. No scheduling algorithm, experiment result value, figure data, or retained CSV content was changed.

## Third-party data decision

The workflow source was identified as the archived `wfcommons/pegasus-instances` project. The audited copy and the upstream repository root did not provide an explicit redistribution license. Therefore no workflow JSON is included. Download scripts pin the upstream source to tag `v1.4` / commit `813a2a7d3e7273200805e89f5475f9126d903eab`, and `manifests/workflow_sha256.txt` verifies the referenced files.

## Safety checks

- Selected text files were scanned for credential-related keywords and private-key markers; no matches were found.
- Selected files were scanned for local absolute paths; no local absolute paths remain.
- No workflow JSON, cache directory, IDE directory, temporary-run directory, or source `.git` metadata remains in the release tree.
- `python -m compileall` completed for `src`, `experiments`, `plots`, and `scripts`.
- A static smoke test completed successfully using a temporary non-retained copy of one workflow instance. It exercised workflow loading, HEFT construction, the event-driven simulator, energy/resource models, and CSV output; temporary files were removed immediately afterward.

## Publication status

GitHub CLI is not installed in the current environment. Consequently no public GitHub repository, remote URL, public release, tag push, Zenodo deposit, DOI, or paper-LaTeX availability statement was created. This avoids fabricating publication identifiers.
