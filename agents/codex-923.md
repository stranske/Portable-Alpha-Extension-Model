<!-- bootstrap for codex on issue #923 -->

## PR Tasks and Acceptance Criteria

**Progress:** 6/6 tasks complete, 0 remaining

### Scope
Current RNG infrastructure has:
- `spawn_rngs(seed, n)` for independent streams
- `spawn_agent_rngs(seed, agent_names)` for per-agent RNGs

The problem: `spawn_agent_rngs` is order-sensitive. If agent ordering changes (or a sleeve is added),
the "same seed" run produces different sleeve randomness. This breaks reproducibility in ways users
interpret as "betrayal" - they expect same seed = same results.

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Modify `spawn_agent_rngs` to derive substreams by stable hashing of `(seed, agent_name)` pairs
- [x] Add deterministic sorting of agent names before stream assignment
- [x] Document the naming contract for substream derivation
- [x] Persist seed + derived substream IDs in output workbook metadata
- [x] Add backward compatibility flag for legacy order-dependent behavior
- [x] Write tests verifying order-independence

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] Adding a new sleeve to a config does not change RNG streams for existing sleeves
- [x] Reordering sleeves in config produces identical results to original order
- [x] Output workbook contains `rng_seed` and `substream_ids` metadata
- [x] Unit tests pass for: order permutations, sleeve additions, seed stability
- [ ] `ruff check` and `mypy` pass
