<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Reduce duplicated pipeline logic across entrypoints by making `pa_core/facade.py` the canonical pipeline and delegating entrypoints to it.

#### Tasks
- [x] Audit all four files to map which pipeline stages are duplicated
- [x] Designate `facade.py` as the canonical pipeline implementation
- [x] Define scope for: Refactor `__main__.py` to call `facade.run_single()` (verify: confirm completion in repo)
- [x] Implement focused slice for: Refactor `__main__.py` to call `facade.run_single()` (verify: confirm completion in repo)
- [x] Validate focused slice for: Refactor `__main__.py` to call `facade.run_single()` (verify: confirm completion in repo)
- [x] Define scope for: Refactor `__main__.py` to call `facade.run_sweep()` (verify: confirm completion in repo)
- [x] Implement focused slice for: Refactor `__main__.py` to call `facade.run_sweep()` (verify: confirm completion in repo)
- [x] Validate focused slice for: Refactor `__main__.py` to call `facade.run_sweep()` (verify: confirm completion in repo)
- [x] Define scope for: Refactor `cli.py` to delegate to `facade.run_single()` (verify: confirm completion in repo)
- [x] Implement focused slice for: Refactor `cli.py` to delegate to `facade.run_single()` (verify: confirm completion in repo)
- [x] Validate focused slice for: Refactor `cli.py` to delegate to `facade.run_single()` (verify: confirm completion in repo)
- [ ] Define scope for: Refactor `cli.py` to delegate to `facade.run_sweep()` (verify: confirm completion in repo)
- [ ] Implement focused slice for: Refactor `cli.py` to delegate to `facade.run_sweep()` (verify: confirm completion in repo)
- [ ] Validate focused slice for: Refactor `cli.py` to delegate to `facade.run_sweep()` (verify: confirm completion in repo)
- [ ] Define scope for: Refactor `cli.py` to delegate to `facade.export()` (verify: confirm completion in repo)
- [ ] Implement focused slice for: Refactor `cli.py` to delegate to `facade.export()` (verify: confirm completion in repo)
- [ ] Validate focused slice for: Refactor `cli.py` to delegate to `facade.export()` (verify: confirm completion in repo)
- [ ] Define scope for: Refactor `orchestrator.py` to call `facade.run_single()` (verify: confirm completion in repo)
- [ ] Implement focused slice for: Refactor `orchestrator.py` to call `facade.run_single()` (verify: confirm completion in repo)
- [ ] Validate focused slice for: Refactor `orchestrator.py` to call `facade.run_single()` (verify: confirm completion in repo)
- [ ] Define scope for: Refactor `orchestrator.py` to call `facade.run_sweep()` (verify: confirm completion in repo)
- [ ] Implement focused slice for: Refactor `orchestrator.py` to call `facade.run_sweep()` (verify: confirm completion in repo)
- [ ] Validate focused slice for: Refactor `orchestrator.py` to call `facade.run_sweep()` (verify: confirm completion in repo)
- [ ] Add deprecation warnings for direct use of non-canonical entry points

#### Acceptance criteria
- [ ] All entry points ultimately call the same pipeline implementation
- [ ] No duplicated pipeline stage logic across files
- [ ] Existing CLI commands should produce the same output and behavior as before the refactor
- [ ] Unit tests should be created or updated to ensure all entry points are tested
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- _None._

<!-- auto-status-summary:end -->
