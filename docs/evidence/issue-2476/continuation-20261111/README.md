# PR #2498 continuation evidence — 2026-11-11

The pre-edit acceptance outcome was that pandas would retain its native all-Series
construction and duplicate-label error, mixed Series/dict rows with duplicate
Series labels would raise before dictionary conversion, and mixed rows with unique
labels would retain their existing records. The focused limit was 120 seconds per
invocation. Each hosted revision and coverage configuration gets one attempt
with a 180-second test limit.

## Focused proof

Repository-selected Python 3.14 environment available on this host was
`/private/tmp/paemv/bin/python` (3.14.3, pandas 3.1.1, pytest 9.2.0).
The named regression command, with that interpreter as `python`:

```sh
timeout -k 5s 120s python -m pytest -q tests/test_coverage_round_2476_facade.py -k 'duplicate_label' -m 'not slow and not live_llm'
```

The production mutation replaced the repaired facade with the exact file from
`496cf6c7884ada74bc1e0586075af28d91a3d486`, reversing
[the facade fix](facade-fix.diff). Both named regressions printed
`DID NOT RAISE` in [red](red.txt). Restoring the repaired bytes produced
[matching SHA-256 values](hashes.txt) and [two passing regressions](green.txt).

```sh
timeout -k 5s 120s python -m pytest -q tests/test_coverage_round_2381.py tests/test_coverage_round_2476_facade.py tests/test_coverage_round_2476_evidence.py tests/test_coverage_round_2476_mutation_transcripts.py -m 'not slow and not live_llm'
```

[Focused acceptance](focused.txt) printed 80 passed and four warnings in 1.68 seconds.

GitHub keepalive advanced the existing branch to
`913ae5cdd655459d25604390403c22a258cc82bf` during the focused work. The
continuation fast-forwarded to retain its commits. The same command then
[printed 87 passed and four warnings in 1.92 seconds](focused-current.txt).
