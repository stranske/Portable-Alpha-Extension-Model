# Portable Alpha-Extension Model

Portable Alpha + Active Extension Model Specification
Below is a comprehensive description of the updated portable‐alpha + active‐extension model, ready to paste into a Markdown cell. Every section is clearly labeled, and all equations use LaTeX delimiters.

## Setup

Run the setup script to create a Python virtual environment and install dependencies:

```bash
./setup.sh
```

Execute this once before running any notebooks or other scripts.

After setting up the environment you can run the command line interface. The
main entry point is ``pa_core.cli`` which exposes additional export and dashboard
options:

```bash
# CSV parameters
python -m pa_core.cli --params parameters.csv --index sp500tr_fred_divyield.csv

# or YAML configuration
python -m pa_core.cli --config params.yaml --index sp500tr_fred_divyield.csv

# optional pivot-style output
python -m pa_core.cli --config params.yaml --index sp500tr_fred_divyield.csv --pivot
```

This writes results to `Outputs.xlsx` in the current directory.

Sample configuration templates live in the `config/` directory.
Both `parameters_template.csv` and `params_template.yml` list all
supported fields and include the mandatory `ShortfallProb` metric.
Copy one of these files to start your own runs.

### Config validation

The helper `pa_core.config.load_config` validates that your
`risk_metrics` list includes all required entries. Missing metrics
such as `ShortfallProb` trigger a `ConfigError` during loading.

> **Warning**
> Large values for `N_SIMULATIONS` or using a very small `External step size (%)` drastically increase runtime. For quick tests, try `N_SIMULATIONS=100` and `External step size (%)=5`.


1. Purpose and High-Level Overview
Goal:
Construct a Monte Carlo framework that allocates a fixed pool of capital (e.g. $1 b) across three “sleeves” (Internal, External Portable-Alpha, and Active Extension), simulates joint returns on Index, In-House α, Extension α, and External PA α, and then reports portfolio metrics (annual return, volatility, VaR, tracking error, breach probability).

Key innovations vs. a simpler portable-alpha model:

Separate “reference period” used to compute index volatility σₙ, which in turn determines the cash/margin needed to synthetically hold 1:1 index exposure.
Three explicit buckets whose dollar-amounts sum to $ 1 b, avoiding any double-counting of β + α exposures.
Active Extension bucket that can be “150/50” or “170/70” long/short, specified by an “Active share (%)” input. By default, we assume 150/50 (i.e. Active share = 50 %) unless the user overrides.
Everything ultimately flows into a set of formulas—one per bucket—that map monthly draws of [ (r_{\beta},,r_{H},,r_{E},,r_{M}) \quad\text{and}\quad f_t ] into portfolio returns.

2. Core Assumptions and Variables
Index (β) returns

We load a historical time series of monthly total returns on the S&P 500 TR (or whichever index) from a CSV.
We partition that series into:
A reference window (e.g. 2010 – 2014) used to compute “reference volatility” σₙ.
An analysis window (e.g. 2015 – 2020) used to compute the actual mean (μₙ) and volatility (σₙ) that drive our Monte Carlo draws.
Three α-streams (simulated jointly with β)

In-House α (
):
Mean = μ_H/12
Vol = σ_H / √12
Correlation ρ_{β,H} with β.
Extension α (
):
Mean = μ_E/12
Vol = σ_E / √12
Correlation ρ_{β,E} with β.
External PA α (
):
Mean = μ_M/12
Vol = σ_M / √12
Correlation ρ_{β,M} with β.
Financing spread (
)

A month-by-month random draw around a drift (financing_mean/12) with vol (financing_vol/12) and occasional jumps of size (spike_factor × (financing_vol/12)), happening with probability spike_prob.
In each month, any bucket that holds ((r_{\beta} − f_t)) is charged that financing cost.
Total fund capital (in millions, default = 1000)

We allocate exactly $ 1 b across three buckets (plus any residual “cash-leftover” after margin).
Standard-deviation multiple (sd_of_vol_mult, default = 3)

“To hold ₙ
 1 b.”
That cash is the internal beta-backing or “margin cash,” needed for futures/swaps.
Three capital buckets (all in $ mm, must sum to 1000)

External PA capital (
)
Manager takes 
 X m of index (β) and ((external_pa_alpha_frac × X m)) of α.
Default α fraction = 50 % ((\theta_{\mathrm{ExtPA}}=0.50)).
Active Extension capital (
)
Manager runs a long/short portfolio with Active share (S).
By default, “150/50” means (S=0.50) (i.e. 150 % long, 50 % short → net 100 %).
Internal PA capital (
)
Runs in-house α; the remainder of internal cash (beyond margin) is used here.
Internal beta backing (
) (computed, not user-entered)
[ W = \sigma_{\text{ref}} \times (\mathrm{sd_of_vol_mult}) \times 1000 \quad (\text{$ mm}). ]

That cash sits in reserve to back a $ 1 b index position via futures/swaps.
Because the external PA and active-extension managers each hold index exposure “inside” their 
 Y m, you do not hold margin for that portion. You only hold (W) for the total $ 1 b.
3. Capital-Allocation Equations
Check:
[ X + Y + Z ;=; 1000 \quad(\text{$ mm}), ]
where

(X = \text{external_pa_capital},)
(Y = \text{active_ext_capital},)
(Z = \text{internal_pa_capital}.)
Margin (internal beta backing):
[ W = \sigma_{\text{ref}} \times (\mathrm{sd_of_vol_mult}) \times 1000 \quad (\text{$ mm}). ]

Internal cash leftover (runs In-House PA):
[ \text{internal_cash_leftover} = 1000 - W - Z \quad (\text{$ mm}). ]

If (W + Z > 1000), the capital structure is infeasible (you cannot hold margin + in-house PA + external buckets all on $ 1 b).
4. Return Equations
We simulate, for each month (t):

[ (r_{\beta,t},,r_{H,t},,r_{E,t},,r_{M,t}) ;\sim;\text{MVN}\bigl([\mu_{\beta},,\mu_H,,\mu_E,,\mu_M],,\Sigma\bigr), ] with

(\mu_{\beta} = \mu_{\text{idx}}) (monthly mean from analysis window),
(\mu_H = \frac{\mu_H^{(\text{annual})}}{12}),
(\mu_E = \frac{\mu_E^{(\text{annual})}}{12}),
(\mu_M = \frac{\mu_M^{(\text{annual})}}{12}).
Covariance (\Sigma) built from:

(\sigma_{\beta} = \sigma_{\text{ref}}) (monthly vol from reference window),
(\sigma_H = \sigma_H^{(\text{annual})}/\sqrt{12}),
(\sigma_E = \sigma_E^{(\text{annual})}/\sqrt{12}),
(\sigma_M = \sigma_M^{(\text{annual})}/\sqrt{12}),
Pairwise correlations (\rho_{\beta,H},,\rho_{\beta,E},,\rho_{\beta,M},,\rho_{H,E},,\dots).
Additionally, each month we draw a financing cost: [ f_t = \frac{\text{financing_mean}}{12} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}\bigl(0,;(\tfrac{\text{financing_vol}}{12})^2\bigr), ] with probability (\text{spike_prob}) of a jump (=\text{spike_factor} \times \frac{\text{financing_vol}}{12}).

4.1. Base (All In-House) Strategy
[ R_{\text{Base},t} = ; (r_{\beta,t} - f_t),\times,w_{\beta_H} ;+; r_{H,t},\times,w_{\alpha_H}. ] By default, (w_{\beta_H} = 0.50) and (w_{\alpha_H} = 0.50).

4.2. External PA Strategy
Capital allocated: (X = \text{external_pa_capital}).
Manager buys X m of index (β) and allocates (\theta_{\mathrm{ExtPA}} = \text{external_pa_alpha_frac}) of that
 X m to α.
Return formula: [ R_{\text{ExtPA},t} = \underbrace{\frac{X}{1000}}{w{\beta}^{\text{ExtPA}}},(r_{\beta,t} - f_t) ;+;\underbrace{\tfrac{X}{1000} ,\times,\theta_{\mathrm{ExtPA}}}{w{\alpha}^{\text{ExtPA}}};(r_{M,t}). ]

If (\theta_{\mathrm{ExtPA}} = 0.50), then half of $ X m is alpha, half is index.
4.3. Active Extension Strategy
Capital allocated: (Y = \text{active_ext_capital}).
Manager runs a long/short portfolio with Active share (S = \frac{\text{active_share_percent}}{100}).
E.g. 150/50 → (S = 0.50).
170/70 → (S = 0.70).
Return formula: [ R_{\text{ActExt},t} = \underbrace{\frac{Y}{1000}}{w{\beta}^{\text{ActExt}}},(r_{\beta,t} - f_t) ;+;\underbrace{\frac{Y}{1000},\times,S}{w{\alpha}^{\text{ActExt}}};(r_{E,t}). ]

The manager’s long/short is embedded in (r_{E,t}).
4.4. Internal Margin & Internal PA
Because both external PA and active-extension managers hold their own index exposure, on your books you only need to hold margin for a single $ 1 b of index. That is: [ W = \sigma_{\text{ref}} \times (\mathrm{sd_of_vol_mult}) \times 1000 \quad (\text{$ mm}). ] Then you also decide to run (Z = \text{internal_pa_capital}) in-house PA:

Internal Beta (margin):
[ R_{\text{IntBet},t} = \Bigl(\tfrac{W}{1000}\Bigr),(r_{\beta,t} - f_t). ]
Internal PA alpha:
[ R_{\text{IntPA},t} = \Bigl(\tfrac{Z}{1000}\Bigr),(r_{H,t}). ]
Internal cash leftover:
[ \text{internal_cash_leftover} = 1000 - W - Z \quad (\text{if positive, earns 0}). ]
5. Putting It All Together in Simulation
Read user inputs (via load_parameters()):

Dates: start_date, end_date, ref_start_date, ref_end_date
Vol/risk: sd_of_vol_mult
Returns: financing_mean, financing_vol, μ_H, σ_H, μ_E, σ_E, μ_M, σ_M
Correlations: ρ_{β,H}, ρ_{β,E}, ρ_{β,M}, ρ_{H,E}, ρ_{H,M}, ρ_{E,M}
Capital buckets: external_pa_capital, external_pa_alpha_frac, active_ext_capital, active_share_percent, internal_pa_capital
Total fund capital (mm): default = 1000
Load index CSV → idx_full (monthly total returns).

Filter

idx_series = idx_full[ start_date : end_date ] → used for μ_β and σ_β.
idx_ref = idx_full[ ref_start_date : ref_end_date ] → used for σ_ref.
Compute
[ \mu_{\beta} = \mathrm{mean}(idx_series), \quad \sigma_{\beta} = \mathrm{std}(idx_series), \quad \sigma_{\text{ref}} = \mathrm{std}(idx_ref). ]

Margin-backing
[ W = \sigma_{\text{ref}} \times \mathrm{sd_of_vol_mult} \times 1000. ] If (W + Z > 1000), error. Else compute [ \text{internal_cash_leftover} = 1000 - W - Z. ]

Build covariance matrix (\Sigma) for ((r_{\beta}, r_H, r_E, r_M)) using
(\sigma_{\beta} = \sigma_{\text{ref}},; \sigma_H = \frac{\sigma_H^{(\text{annual})}}{\sqrt{12}},; \sigma_E = \frac{\sigma_E^{(\text{annual})}}{\sqrt{12}},; \sigma_M = \frac{\sigma_M^{(\text{annual})}}{\sqrt{12}},)
and correlations.

Monte Carlo draws:
For each of (N_{\text{SIMULATIONS}}) trials, simulate a (T=N_{\text{MONTHS}})-month path of (,(r_{\beta,t},,r_{H,t},,r_{E,t},,r_{M,t})) and financing (f_t).

Compute monthly returns for each bucket:

Base:
[ R_{\text{Base},t} = (r_{\beta,t} - f_t),w_{\beta_H} ;+; r_{H,t},w_{\alpha_H}. ]
External PA:
[ R_{\text{ExtPA},t} = \bigl(\tfrac{X}{1000}\bigr)(r_{\beta,t} - f_t) ;+; \bigl(\tfrac{X}{1000},\theta_{\mathrm{ExtPA}}\bigr)(r_{M,t}). ]
Active Extension:
[ R_{\text{ActExt},t} = \bigl(\tfrac{Y}{1000}\bigr)(r_{\beta,t} - f_t) ;+; \bigl(\tfrac{Y}{1000},S\bigr)(r_{E,t}). ]
Internal Beta:
[ R_{\text{IntBet},t} = \bigl(\tfrac{W}{1000}\bigr)(r_{\beta,t} - f_t). ]
Internal PA α:
[ R_{\text{IntPA},t} = \bigl(\tfrac{Z}{1000}\bigr)(r_{H,t}). ]
Note: We only report three portfolios—“Base,” “ExternalPA,” and “ActiveExt.” Each one compounds its own monthly returns for a 12-month horizon: [ R_{\text{bucket}}^{\text{(year)}} = \prod_{t=1}^{12} (1 + R_{\text{bucket},t}) - 1. ]

Compute performance metrics for each portfolio’s annual returns:

Ann Return = sample mean.
Ann Vol = sample standard deviation.
VaR 95% = 5th percentile.
Tracking Error = std of (bucket_return − index_return).
Breach Probability = % of months (in the first sim path) where ((r_{\text{bucket},t} < -,\mathrm{buffer_multiple}\times\sigma_{\beta})).
Export

Inputs sheet: all parameters (dates, vol caps, bucket sizes, α fractions, active share, σ_ref, W, internal cash leftover, etc.).
Summary sheet: metrics for “Base,” “ExternalPA,” and “ActiveExt.”
Raw returns sheets: monthly paths for each bucket (first simulation) so users can inspect breach months.
6. Input Parameters Summary
Below is a consolidated list of every input variable that must appear in the “friendly” CSV:

Date ranges

Start date → start_date (analysis window begin).
End date → end_date (analysis window end).
Reference start date → ref_start_date (for σ_ref).
Reference end date → ref_end_date (for σ_ref).
Financing parameters

Annual financing mean (%) → financing_mean_annual (default = 0.50 %).
Annual financing vol (%) → financing_vol_annual (default = 0.10 %).
Monthly spike probability → spike_prob (default = 2 %).
Spike size (σ × multiplier) → spike_factor (default = 2.25).
In-House PA parameters

In-House annual return (%) → mu_H (default = 4.00 %).
In-House annual vol (%) → sigma_H (default = 1.00 %).
In-House β → w_beta_H (default = 0.50).
In-House α → w_alpha_H (default = 0.50).
Extension α parameters

Alpha-Extension annual return (%) → mu_E (default = 5.00 %).
Alpha-Extension annual vol (%) → sigma_E (default = 2.00 %).
Active Extension capital (mm) → active_ext_capital (default = 0).
Active share (%) → active_share_percent (default = 50 % ⇒ a 150/50 program).
External PA α parameters

External annual return (%) → mu_M (default = 3.00 %).
External annual vol (%) → sigma_M (default = 2.00 %).
External PA capital (mm) → external_pa_capital (default = 0).
External PA α fraction (%) → external_pa_alpha_frac (default = 50 %).
Correlations

Corr index–In-House → rho_idx_H (default = 0.05).
Corr index–Alpha-Extension → rho_idx_E (default = 0.00).
Corr index–External → rho_idx_M (default = 0.00).
Corr In-House–Alpha-Extension → rho_H_E (default = 0.10).
Corr In-House–External → rho_H_M (default = 0.10).
Corr Alpha-Extension–External → rho_E_M (default = 0.00).
Capital & risk backing

Total fund capital (mm) → total_fund_capital (default = 1000).
Standard deviation multiple → sd_of_vol_mult (default = 3).
Internal PA capital (mm) → internal_pa_capital (default = 0).
Buffer multiple → buffer_multiple (default = 3).
Legacy/Optional

X grid (mm) → X_grid_list (list of X values).
External manager α fractions → EM_thetas_list.
7. Output Considerations
Inputs sheet (Excel):
List every single parameter, including:

Date windows (analysis and reference),
Financing parameters,
α-stream parameters,
Correlations,
Capital buckets (X, Y, Z),
SD multiple, margin backing (W), internal cash leftover,
Active share, etc.
Summary sheet (Excel):
For each portfolio (“Base,” “ExternalPA,” “ActiveExt”), show:

Annual Return (%),
Annual Volatility (%),
95 % VaR (%),
Tracking Error (%),
Breach Probability (%).
Raw returns sheets (Excel):
Monthly paths for each bucket (first simulation), so users can inspect “breach” months where (R_{t} < -(\text{buffer_multiple} × σ_{\beta})).

Console output:
A “human‐friendly” summary, e.g.:

For “ExternalPA (X = 300, 50 % α)”:
• Expected annual return: 10.2 %
• Annual volatility: 12.3 %
• 95 % VaR: −3.4 %
• Tracking error: 8.7 %
• Breach probability: 2.0 %.

8. Intuition Behind Key Pieces
Why a separate reference period?

If you measure index volatility over the same window you analyze (e.g. 2015–2020), you capture “current regime” vol. Often, managers prefer a longer/different window (e.g. 2010–2014) to gauge typical funding volatility. That reference σₙ, times a multiple (e.g. 3×), tells you how much cash to set aside to back $ 1 b of index exposure.
Why Active share as a percentage?

A “150/50” program has 150 % long and 50 % short = net 100 %. Its “active share” is reported as 50 %.
If you want “170/70,” then active share = 70 %.
The code converts “Active share (%)” to decimal (S). For a 150/50 program, the default is 50 % ((S = 0.50)).
Why each bucket’s formula ensures no double-counting

Whenever you give $ X m to External PA, that manager holds the index exposure on your behalf. You do not hold margin for that portion. Similarly, the Active Extension manager holds their own index.
On your books, you only need to hold margin for a single $ 1 b index. That is (W).
Once you hand 
 Y m to active ext, both managers hold ((X + Y)) of index on your behalf. So your margin (W) backs the entire $ 1 b, not just the “leftover” portion.
9. Step-by-Step Implementation Checklist
Read and parse user parameters (dates, vols, α fractions, active share, capital buckets, etc.).

Load index CSV → idx_full.

Filter → idx_ref for σ_ref; idx_series for μ_β and σ_β.

Compute:
[ μ_β = \mathrm{mean}(idx_series), \quad σ_β = \mathrm{std}(idx_series), \quad σ_{\text{ref}} = \mathrm{std}(idx_ref). ]

Margin-backing:
[ W = σ_{\text{ref}} × (\mathrm{sd_of_vol_mult}) × 1000. ] Check (W + Z ≤ 1000). Compute leftover internal cash = (1000 - W - Z).

Build covariance matrix using ((σ_{\text{ref}},,σ_H/√{12},,σ_E/√{12},,σ_M/√{12})) plus correlations.

Monte Carlo draws:
For each of (N_{\mathrm{SIM}}) trials, simulate a path of length (T = N_{\mathrm{MONTHS}}) for ((r_{\beta,t},,r_{H,t},,r_{E,t},,r_{M,t})) and financing (f_t).

Compute monthly returns:

Base:
[ R_{\text{Base},t} = (r_{\beta,t} - f_t),w_{\beta_H} + r_{H,t},w_{\alpha_H}. ]
External PA:
[ R_{\text{ExtPA},t} = \Bigl(\tfrac{X}{1000}\Bigr)(r_{\beta,t} - f_t) ;+;\Bigl(\tfrac{X}{1000},\theta_{\mathrm{ExtPA}}\Bigr)(r_{M,t}). ]
Active Extension:
[ R_{\text{ActExt},t} = \Bigl(\tfrac{Y}{1000}\Bigr)(r_{\beta,t} - f_t) ;+;\Bigl(\tfrac{Y}{1000},S\Bigr)(r_{E,t}). ]
Internal Beta:
[ R_{\text{IntBet},t} = \Bigl(\tfrac{W}{1000}\Bigr)(r_{\beta,t} - f_t). ]
Internal PA α:
[ R_{\text{IntPA},t} = \Bigl(\tfrac{Z}{1000}\Bigr)(r_{H,t}). ]
Aggregate monthly → annual returns for “Base,” “ExternalPA,” “ActiveExt.”

Compute metrics:

Ann Return, Ann Vol, VaR 95, Tracking Error, Breach Probability.
Export Inputs, Summary, Raw returns to Excel + print narrative.
