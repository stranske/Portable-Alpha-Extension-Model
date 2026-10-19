# PAEM baseline coverage manifest

- Input parameters: **103**
- Exercised by a scenario: **10** (9.7%)
- Observed read at runtime: **60**

## Priority parameters

- [x] `sigma_H`
- [x] `mu_H`
- [x] `rho_idx_H`
- [x] `internal_pa_financing_mean_month`
- [x] `active_share`

## Evidence boundary

Reads are recorded only around the real run_single call after validation.
Bound model_dump serialization and later report traversal do not add reads.
A read is access evidence, not numerical wiring proof. Enforced scenarios
below passed directional comparisons against the same-seed funded control.
Correlation scenarios vary six fields together: their proof is joint,
not independent influence for each individual correlation.

| Input | Catalog touched | Computational read | Behavioral evidence |
| --- | --- | --- | --- |
| `active_share` | yes | not observed | unproven / report-only |
| `internal_pa_financing_mean_month` | yes | yes | enforced directional scenario |
| `mu_H` | yes | yes | enforced directional scenario |
| `rho_E_M` | yes | yes | enforced directional scenario |
| `rho_H_E` | yes | yes | enforced directional scenario |
| `rho_H_M` | yes | yes | enforced directional scenario |
| `rho_idx_E` | yes | yes | enforced directional scenario |
| `rho_idx_H` | yes | yes | enforced directional scenario |
| `rho_idx_M` | yes | yes | enforced directional scenario |
| `sigma_H` | yes | yes | enforced directional scenario |

`internal_pa_financing_mean_month` funds InternalPA; the separate
`internal_financing_mean_month` is the margin-financing input and is not
substituted into that sleeve's directional claim.
`active_share` remains report-only in this returns-mode scenario. Its
validation/derived-agent construction is outside this runtime attribute
recorder; absence of a read is not proof that the input is globally ignored.
This bounded baseline is coordinated with broad coverage issue #2336;
it makes no repository-wide coverage or universal parameter-influence claim.
