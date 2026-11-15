# Issue #2507 Verification Evidence

## Scope

Follow-up to [#2504](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2504) and merged commit `f8f695111c4fb11a0ee97e749f8f7fca1e1af74b` ([PR #2505](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2505)):
1. Execute `python -m pytest tests/test_schema_export.py -q` under provisioned Python 3.13 and Python 3.14, including `test_parameter_dictionary_matches_schema`.
2. Run `python -m pa_core.schema --generate-parameter-dictionary` under both interpreters, compare `docs/guides/PARAMETER_DICTIONARY.md` byte-for-byte, and commit any generated change.
3. Address the two residual concerns noted in the PR #2505 verification report:
   - Sort imports in `tests/test_schema_export.py` (`isort` check failure in `Python CI / lint-format`).
   - Normalize nested generic type arguments (such as `Dict[str, Optional[FeeSchedule]]` vs `Dict[str, FeeSchedule | None]` and `List[Path]`) in `pa_core/schema.py::_stringify_type` and add cross-interpreter regression coverage in `tests/test_schema_export.py`.

## Provisioned Interpreters

| Interpreter | Path | Version | Key Packages |
| --- | --- | --- | --- |
| Python 3.13 | `/opt/hostedtoolcache/Python/3.13.18/x64/bin/python` | `Python 3.13.18` | `pytest==9.2.0`, `pydantic==2.14.3`, `pydantic-core==2.50.4`, `PyYAML==6.0.3` |
| Python 3.14 | `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python` | `Python 3.14.10` | `pytest==9.2.0`, `pydantic==2.14.3`, `pydantic-core==2.50.4`, `PyYAML==6.0.3` |

## Task 1: `pytest tests/test_schema_export.py` under Python 3.13 and 3.14

### Quiet runs (`-q`)

```text
$ /opt/hostedtoolcache/Python/3.13.18/x64/bin/python -m pytest tests/test_schema_export.py -q
......................                                                   [100%]
22 passed in 2.38s

$ /opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pytest tests/test_schema_export.py -q
......................                                                   [100%]
22 passed, 13 warnings in 2.53s
```

### Verbose verification of `test_parameter_dictionary_matches_schema` and `test_parameter_dictionary_byte_identical_across_313_and_314`

Both interpreters execute and pass `test_parameter_dictionary_matches_schema` and `test_parameter_dictionary_byte_identical_across_313_and_314` (along with all 20 other test cases in `tests/test_schema_export.py`).

## Task 2: Byte-for-byte comparison of `docs/guides/PARAMETER_DICTIONARY.md`

```text
$ /opt/hostedtoolcache/Python/3.13.18/x64/bin/python -m pa_core.schema --generate-parameter-dictionary --dictionary-output /tmp/PARAMETER_DICTIONARY_313.md
$ /opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pa_core.schema --generate-parameter-dictionary --dictionary-output /tmp/PARAMETER_DICTIONARY_314.md
$ /opt/hostedtoolcache/Python/3.13.18/x64/bin/python -m pa_core.schema --generate-parameter-dictionary
$ /opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pa_core.schema --generate-parameter-dictionary
$ cmp -l /tmp/PARAMETER_DICTIONARY_313.md /tmp/PARAMETER_DICTIONARY_314.md
$ cmp -l /tmp/PARAMETER_DICTIONARY_313.md docs/guides/PARAMETER_DICTIONARY.md
$ sha256sum /tmp/PARAMETER_DICTIONARY_313.md /tmp/PARAMETER_DICTIONARY_314.md docs/guides/PARAMETER_DICTIONARY.md
0e50861d668127458f702be7453db3f7fbbe8c70a283a8851334a7d98ea91b3f  /tmp/PARAMETER_DICTIONARY_313.md
0e50861d668127458f702be7453db3f7fbbe8c70a283a8851334a7d98ea91b3f  /tmp/PARAMETER_DICTIONARY_314.md
0e50861d668127458f702be7453db3f7fbbe8c70a283a8851334a7d98ea91b3f  docs/guides/PARAMETER_DICTIONARY.md
$ wc -c /tmp/PARAMETER_DICTIONARY_313.md /tmp/PARAMETER_DICTIONARY_314.md docs/guides/PARAMETER_DICTIONARY.md
12454 /tmp/PARAMETER_DICTIONARY_313.md
12454 /tmp/PARAMETER_DICTIONARY_314.md
12454 docs/guides/PARAMETER_DICTIONARY.md
37362 total
$ git diff docs/guides/PARAMETER_DICTIONARY.md
(empty)
```

Why `docs/guides/PARAMETER_DICTIONARY.md` has zero diff:
- Commit `37032da53a48d6137887119bc000e900d6aa167a` (#2434) already regenerated `docs/guides/PARAMETER_DICTIONARY.md` under Python 3.14 using the canonical `X | None` and `pathlib.Path | None` spellings (`12,454` bytes, SHA-256 `0e50861d668127458f702be7453db3f7fbbe8c70a283a8851334a7d98ea91b3f`).
- With `pa_core/schema.py::_stringify_type` normalizing `typing.Union`/`Optional` and private module paths (`pathlib._local.Path` -> `pathlib.Path`), both Python 3.13.18 and Python 3.14.10 generate that exact `12,454`-byte file (`0e50861d668127458f702be7453db3f7fbbe8c70a283a8851334a7d98ea91b3f`), leaving no uncommitted or changed bytes in `docs/guides/PARAMETER_DICTIONARY.md`.
