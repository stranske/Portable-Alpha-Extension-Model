<!-- bootstrap for codex on issue #865 -->

## PR Tasks and Acceptance Criteria

**Progress:** 15/15 tasks complete, 0 remaining

### Scope
Run outputs are scattered across multiple files and conventions. A first-class "Run Artifact" bundle would guarantee reproducibility, make sharing easier, and provide a stable interface for downstream consumers (dashboard, reports, audits).

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Define `RunArtifact` dataclass with required fields: config, index_hash, seed, manifest, outputs
- [x] Create `RunArtifactBundle` class that manages artifact directory
- [x] Implement `bundle.save(path)` to write artifact bundle to disk
- [x] Implement `bundle.load(path)` to read artifact bundle from disk
- [x] Implement `bundle.verify()` to check bundle integrity (hashes match, required files present)
- [x] Add config snapshot (exact YAML used, not resolved) to bundle
- [x] Add index series hash to bundle for reproducibility verification
- [x] Add random seed to bundle
- [x] Update CLI to produce artifact bundles
- [x] Add `--bundle` output option to CLI

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] Artifact bundle contains all required components
- [x] Bundle can be loaded and verified after creation
- [x] Bundle verification fails if any component is modified
- [x] Bundle includes enough information to reproduce the run
- [x] Existing output format remains available (bundle is opt-in initially)
