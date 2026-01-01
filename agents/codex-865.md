<!-- bootstrap for codex on issue #865 -->

## PR Tasks and Acceptance Criteria

**Progress:** 1/15 tasks complete, 14 remaining

### Scope
Run outputs are scattered across multiple files and conventions. A first-class "Run Artifact" bundle would guarantee reproducibility, make sharing easier, and provide a stable interface for downstream consumers (dashboard, reports, audits).

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Define `RunArtifact` dataclass with required fields: config, index_hash, seed, manifest, outputs
- [ ] Create `RunArtifactBundle` class that manages artifact directory
- [ ] Implement `bundle.save(path)` to write artifact bundle to disk
- [ ] Implement `bundle.load(path)` to read artifact bundle from disk
- [ ] Implement `bundle.verify()` to check bundle integrity (hashes match, required files present)
- [ ] Add config snapshot (exact YAML used, not resolved) to bundle
- [ ] Add index series hash to bundle for reproducibility verification
- [ ] Add random seed to bundle
- [ ] Update CLI to produce artifact bundles
- [ ] Add `--bundle` output option to CLI

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [ ] Artifact bundle contains all required components
- [ ] Bundle can be loaded and verified after creation
- [ ] Bundle verification fails if any component is modified
- [ ] Bundle includes enough information to reproduce the run
- [ ] Existing output format remains available (bundle is opt-in initially)
