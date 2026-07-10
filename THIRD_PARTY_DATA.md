# Third-Party Workflow Data

## Status

This repository does **not** redistribute any workflow JSON files.

The audited source workspace contained copies of workflow execution instances from the archived [WfCommons Pegasus Instances repository](https://github.com/wfcommons/pegasus-instances). Its root directory and public repository page identify the data source and JSON format, but no explicit license file was present in the audited copy or repository root. Redistribution permission was therefore not established for this release.

## Pinned source

- Source URL: `https://github.com/wfcommons/pegasus-instances.git`
- Tag: `v1.4`
- Commit: `813a2a7d3e7273200805e89f5475f9126d903eab`
- Format: WfCommons JSON

Use `scripts/download_workflows.sh` or `scripts/download_workflows.ps1` to clone that exact revision into `data/pegasus-instances`. The scripts intentionally fail if that directory already exists, preventing an implicit replacement of locally obtained data.

## Instances referenced by this release

The checksum manifest covers the Montage, Epigenomics, and Seismology instances used by the public static/dynamic configurations and cross-family study. Run `python scripts/verify_workflows.py` after download. The checksums identify the files used by this artifact; they do not grant a redistribution license.

The public artifact does not claim authorship of these workflow instances. Consult the upstream project and its release materials for data citation and any later licensing information.
