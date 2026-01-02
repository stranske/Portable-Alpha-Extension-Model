<!-- bootstrap for codex on issue #927 -->

## PR Tasks and Acceptance Criteria

**Progress:** 7/11 tasks complete, 4 remaining

### Scope
Pydantic aliases for human-readable CSV/YAML fields are a nice UX trick, but they create fragility:
- Capitalization/punctuation changes break parsing
- Excel "helpfulness" corrupts templates (smart quotes, auto-formatting)
- Drift between docs/templates and actual model field aliases
- Tests have already broken around CSV format expectations

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Create schema export function that outputs all fields + their aliases as JSON/YAML
- [x] Build template generator that creates CSV/YAML templates from schema
- [x] Add CI check that templates match current schema (fail if drift detected)
- [ ] Generate a "parameter dictionary" documentation page from schema
- [x] Add Excel-safe export option (no smart quotes, explicit quoting)
- [ ] Document the aliasing convention in CONTRIBUTING.md

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] `python -m pa_core.schema --export` produces machine-readable field definitions
- [x] Templates in `templates/` are auto-generated (have generation comment header)
- [x] CI fails if templates drift from schema
- [ ] Documentation includes auto-generated parameter reference
- [ ] `ruff check` and `mypy` pass
