# ECDS Artifact v1.0.0

## Included

- ECDS together with HEFT, GREENHEFT, and MOHEFT scheduling implementations.
- An event-driven simulator with energy, renewable-energy, carbon, and resource models.
- Static, dynamic, cross-family, and ablation experiment configurations.
- Fixed release seeds, retained result summaries, and plotting scripts.
- Reproducibility documentation, workflow download scripts, and SHA-256 verification manifests.

## Workflow data

Workflow instances are obtained separately from the WfCommons Pegasus execution-instance repository at commit `813a2a7d3e7273200805e89f5475f9126d903eab`. Workflow JSON files are not redistributed in this artifact. The included scripts and checksums retrieve and verify the required instances.

## Known limitations

The results are from simulation experiments, not a physical cluster deployment or a complete life-cycle carbon assessment. The simulator site profiles and machine parameters are experimental model inputs rather than production-cloud measurements. Results may vary across platforms and dependency versions; retained CSV values are preserved rather than modified to force rerun agreement.
