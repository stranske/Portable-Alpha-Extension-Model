# How much active extension is enough?

This model can answer the central active-extension question — *how large should the active
bet (active share / external-PA alpha fraction) be?* — using a transfer-coefficient
("diminishing returns") control. This guide explains the parameters and walks a worked
example. Background: issue #1924 and the fundamental law of active management
(Grinold–Kahn; Clarke/de Silva/Sapra/Thorley, *"Long/Short Extensions: How Much Is Enough?"*).

## Why a linear lever is wrong

By default (`*_tc_decay = 0`) the active-share lever `s` scales the **whole** extension-alpha
stream — both its mean and its volatility — so the information ratio (return ÷ risk) is
**invariant** to `s`. That implies "more active share is always proportionally more alpha,"
which contradicts the literature: relaxing the long-only constraint raises the **transfer
coefficient** with *diminishing* returns, and beyond some point the marginal alpha no longer
justifies the marginal risk and cost.

## The transfer-coefficient control

Set a positive decay `κ` to make expected alpha **concave** in the lever while active risk
keeps scaling linearly:

> **TC(s) = 1 / (1 + κ·s)**  →  realized IR(s) = (μ/σ) / (1 + κ·s)  (declines with s)
> gross alpha mean(s) = s · μ / (1 + κ·s)  (concave, saturating)

| Parameter | Meaning |
|---|---|
| `active_share_tc_decay` (κ) | Diminishing-returns rate for the **ActiveExt** sleeve. `0` = linear/legacy. |
| `theta_tc_decay` (κ) | Same, for the **ExternalPA** sleeve's `theta_extpa`. |
| `active_ext_cost_per_share` (c) | Optional monthly extension cost per unit of `active_share` (fraction of NAV). |
| `ext_pa_cost_per_share` (c) | Same, for `theta_extpa`. |

With a positive cost, net alpha mean `N(s) = s·μ/(1+κs) − c·s` has a **closed-form interior
optimum**:

> **s\* = ( √(μ/c) − 1 ) / κ**     (where μ is the *monthly* alpha mean of the sleeve's stream)

### Recommended preset — "moderate diminishing returns"
`κ = 0.43` makes the IR at `active_share = 1` about **70%** of the small-bet IR
(`1/(1+0.43) = 0.70`). Start here and calibrate from a manager's realized information ratio
at two active-risk levels (a natural hand-off from the `Trend_Model_Project` manager
evaluation — see the repo audit, item 7).

> Backward compatibility: all four fields default to `0.0`, which reproduces the prior
> linear behaviour **exactly**. Non-zero values are explicit opt-in.

## Worked example

`examples/scenarios/active_extension_diminishing_returns.yml` ships the preset
(`κ = 0.43`, `cost = 0.0026/month`, `mu_E = 5%/yr ⇒ 0.4167%/month`). The closed-form optimum
is `s* = (√(0.004167/0.0026) − 1)/0.43 ≈ 0.62`.

Running the ActiveExt sleeve's annualized return across active-share levels:

| active_share | ActiveExt ann. return |
|---:|---:|
| 0.20 | 4.109% |
| 0.40 | 4.167% |
| 0.50 | 4.182% |
| **0.62** | **4.189%  ← peak** |
| 0.70 | 4.187% |
| 0.80 | 4.180% |
| 1.00 | 4.146% |

The empirical optimum (0.62) matches the closed form — return rises, peaks, then falls, so
the model now identifies an *interior* "right size" instead of always recommending more.

Run it yourself (an `alpha_shares` sweep over active share):

```bash
pa run --config examples/scenarios/active_extension_diminishing_returns.yml \
       --index data/sp500tr_fred_divyield.csv --output Outputs.xlsx
```

Then open the **Results** page (or the Summary sheet) and read the net return across the
swept `active_share` grid — it traces the concave curve above.

## Caveats
- The transfer-coefficient haircut uses the configured **monthly** alpha mean (`mu_E`/`mu_M`).
  Under regime-switching the realized stream mean is a mixture, so the haircut is approximate
  in that mode.
- `κ` and `cost` are assumptions; they should be calibrated (e.g. from realized manager IRs
  and borrow/turnover costs), and like all inputs they drive the result — see the audit's
  "questions to ask before presenting" list.
