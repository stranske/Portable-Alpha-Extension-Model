<!-- bootstrap for codex on issue #1204 -->

## PR Tasks and Acceptance Criteria

**Progress:** 9/9 tasks complete, 0 remaining

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Replace `_SWEEP_CACHE: Dict[str, List[SweepResult]]` with an `OrderedDict` (or equivalent) that preserves insertion/use ordering.
- [x] Add `SWEEP_CACHE_MAX_ENTRIES` constant (e.g., 8) in `pa_core/sweep.py`.
- [x] On cache insert, evict least-recently-used entries until `len(cache) <= SWEEP_CACHE_MAX_ENTRIES`.
- [x] Add `clear_sweep_cache()` function in `pa_core/sweep.py`.
- [x] Update `run_parameter_sweep_cached()` to refresh LRU order on access.
- [x] Existing deterministic caching test still passes (`res1 is res2`).
- [x] New test: exceed cache max entries and assert oldest key is evicted.
- [x] New test: `clear_sweep_cache()` empties cache.
- [x] Tests pass.

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] Cache size never exceeds `SWEEP_CACHE_MAX_ENTRIES`.
- [x] Repeated calls with the same key still return the cached object.
- [x] Eviction removes the least-recently-used entry.
- [x] Tests pass.
