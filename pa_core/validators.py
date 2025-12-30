"""Enhanced validation functions for input guardrails and real-time validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import CORRELATION_LOWER_BOUND, CORRELATION_UPPER_BOUND

# Module-level constants for validation thresholds
#
# These constants define the thresholds used in validation functions.
# Defining them as named constants improves maintainability and makes
# it easier to adjust thresholds without searching for magic numbers.

MIN_RECOMMENDED_STEP_SIZE = 0.1
"""float: Threshold below which step sizes are considered very small.

Step sizes below this threshold may result in excessive parameter combinations
during parameter sweeps, leading to long computation times. A warning is
issued when step sizes fall below this value.
"""

LOW_BUFFER_THRESHOLD = 0.1
"""float: Threshold for capital buffer warnings.

When the available capital buffer falls below this percentage of total capital,
a warning is issued to alert users that capital allocation is approaching limits.
"""

NUMERICAL_STABILITY_EPSILON = 1e-12
"""float: Small epsilon value for numerical stability.

This constant is used to regularize covariance matrices by adding to the diagonal
when they are not positive semi-definite, ensuring numerical stability during
multivariate normal sampling and other matrix operations.
"""

TEST_TOLERANCE_EPSILON = 1e-12
"""float: Very small tolerance for numerical test assertions.

This constant is used as an absolute tolerance when testing that values are 
approximately zero in unit tests, particularly for tracking error calculations.
"""

SYNTHETIC_DATA_MEAN = 0.0
"""float: Mean value for generating synthetic index data.

This constant defines the mean value used when generating synthetic index return
data for testing purposes. Using a zero mean represents a neutral expected return
assumption for test scenarios.
"""

SYNTHETIC_DATA_STD = 0.01
"""float: Standard deviation for generating synthetic index data.

This constant defines the standard deviation used when generating synthetic index
return data for testing purposes. The value of 1% represents a realistic monthly
volatility level for index returns in test scenarios.
"""

VOLATILITY_STRESS_MULTIPLIER = 3
"""int: Default multiplier for volatility stress testing.

This constant represents the multiplier applied to volatility parameters
in stress test scenarios such as the 2008_vol_regime preset, where volatilities
are increased by this factor to simulate high-volatility market conditions.
"""


def select_vol_regime_sigma(
    values: np.ndarray | pd.Series,
    *,
    regime: Literal["single", "two_state"],
    window: int,
) -> tuple[float, str | None, int | None]:
    """Select volatility based on a single or two-state regime."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("volatility series must contain data")
    base_sigma = float(np.std(arr, ddof=1))
    if regime == "two_state":
        if window <= 1:
            raise ValueError("vol_regime_window must be > 1 for two_state regime")
        if arr.size > 1:
            recent = arr[-window:]
            if recent.size > 1:
                recent_sigma = float(np.std(recent, ddof=1))
                state = "high" if recent_sigma >= base_sigma else "low"
                window_used = int(min(window, recent.size))
                return recent_sigma, state, window_used
    return base_sigma, None, None


class ValidationResult(NamedTuple):
    """Result of a validation check."""

    is_valid: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    details: Dict[str, Any] = {}


class PSDProjectionInfo(NamedTuple):
    """Information about PSD projection."""

    was_projected: bool
    max_delta: float
    max_eigenvalue_delta: float
    original_min_eigenvalue: float
    projected_min_eigenvalue: float


def validate_correlations(correlations: Dict[str, float]) -> List[ValidationResult]:
    """Validate correlation values are within bounds [-1, 1].

    Args:
        correlations: Dictionary of correlation names to values

    Returns:
        List of validation results
    """
    results = []

    for name, rho in correlations.items():
        if not (CORRELATION_LOWER_BOUND <= rho <= CORRELATION_UPPER_BOUND):
            results.append(
                ValidationResult(
                    is_valid=False,
                    message=f"Correlation {name}={rho:.3f} is outside valid range [{CORRELATION_LOWER_BOUND}, {CORRELATION_UPPER_BOUND}]",
                    severity="error",
                    details={
                        "parameter": name,
                        "value": rho,
                        "bounds": [CORRELATION_LOWER_BOUND, CORRELATION_UPPER_BOUND],
                    },
                )
            )
        elif abs(rho) > 0.95:
            results.append(
                ValidationResult(
                    is_valid=True,
                    message=f"High correlation {name}={rho:.3f} may lead to numerical instability",
                    severity="warning",
                    details={"parameter": name, "value": rho},
                )
            )

    return results


def validate_covariance_matrix_psd(
    cov_matrix: np.ndarray, label: str = "covariance matrix"
) -> Tuple[ValidationResult, PSDProjectionInfo]:
    """Validate covariance matrix is positive semidefinite and provide projection info.

    Args:
        cov_matrix: Covariance matrix to validate
        label: Human-readable label for the matrix

    Returns:
        Tuple of (validation result, PSD projection info)
    """
    # Lazy import to avoid circular dependency at module import time
    from .sim.covariance import _is_psd, nearest_psd

    if not _is_psd(cov_matrix):
        # Perform projection and gather detailed info
        original_eigenvalues = np.linalg.eigvalsh(cov_matrix)
        projected_matrix = nearest_psd(cov_matrix)
        projected_eigenvalues = np.linalg.eigvalsh(projected_matrix)

        max_delta = float(np.max(np.abs(projected_matrix - cov_matrix)))
        max_eigenvalue_delta = float(np.max(projected_eigenvalues - original_eigenvalues))
        original_min_eig = float(original_eigenvalues.min())
        projected_min_eig = float(projected_eigenvalues.min())

        psd_info = PSDProjectionInfo(
            was_projected=True,
            max_delta=max_delta,
            max_eigenvalue_delta=max_eigenvalue_delta,
            original_min_eigenvalue=original_min_eig,
            projected_min_eigenvalue=projected_min_eig,
        )

        result = ValidationResult(
            is_valid=True,  # Still valid after projection
            message=f"Projected to PSD: Δλmax = {max_eigenvalue_delta:.2e}, max|Δ| = {max_delta:.2e}",
            severity="warning",
            details={
                "projection_info": psd_info._asdict(),
                "original_min_eigenvalue": original_min_eig,
                "max_delta": max_delta,
            },
        )
    else:
        psd_info = PSDProjectionInfo(
            was_projected=False,
            max_delta=0.0,
            max_eigenvalue_delta=0.0,
            original_min_eigenvalue=float(np.linalg.eigvalsh(cov_matrix).min()),
            projected_min_eigenvalue=float(np.linalg.eigvalsh(cov_matrix).min()),
        )

        result = ValidationResult(
            is_valid=True,
            message=f"{label.capitalize()} is positive semidefinite",
            severity="info",
            details={"min_eigenvalue": psd_info.original_min_eigenvalue},
        )

    return result, psd_info


def load_margin_schedule(path: Path) -> pd.DataFrame:
    """Load and validate broker margin schedule.

    The schedule must contain ``term`` and ``multiplier`` columns representing
    the term (in months) and corresponding margin multiplier.  The returned
    frame is sorted by term to support interpolation. Additional validation
    ensures terms are non-negative, multipliers are positive and the term
    structure is strictly increasing to avoid interpolation ambiguities.
    """
    if not path.exists():
        raise FileNotFoundError(f"Margin schedule file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    if "term" not in df.columns and "term_months" in df.columns:
        df = df.rename(columns={"term_months": "term"})
    required_cols = {"term", "multiplier"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Margin schedule CSV file missing required columns: {missing}")
    df["term"] = pd.to_numeric(df["term"], errors="coerce")
    df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce")
    if bool(df[["term", "multiplier"]].isna().any().any()):
        raise ValueError("Margin schedule contains non-numeric or missing term/multiplier values")
    df = df.sort_values("term")

    if bool((df["term"] < 0).any()):
        raise ValueError("Margin schedule terms must be non-negative")
    if bool((df["multiplier"] <= 0).any()):
        raise ValueError("Margin schedule multipliers must be positive")
    if bool(df["term"].duplicated().any()):
        raise ValueError(
            "Margin schedule terms must not contain duplicates (each term value must be unique)"
        )
    if bool(df["term"].diff().dropna().le(0).any()):
        raise ValueError(
            "Margin schedule terms must be strictly increasing (each term must be greater than the previous)"
        )
    return df.sort_values("term")


def interpolate_margin_multiplier(term_months: float, schedule: pd.DataFrame) -> float:
    """Return the interpolated margin multiplier for a given term."""
    terms = schedule["term"].to_numpy(float)
    multipliers = schedule["multiplier"].to_numpy(float)
    if terms.size == 0:
        raise ValueError("Margin schedule must contain at least one row")
    return float(np.interp(float(term_months), terms, multipliers))


def calculate_margin_requirement(
    reference_sigma: float,
    volatility_multiple: float = 3.0,
    total_capital: float = 1000.0,
    *,
    financing_model: str = "simple_proxy",
    margin_schedule: Optional[pd.DataFrame] = None,
    schedule_path: Optional[Path] = None,
    term_months: float = 1.0,
) -> float:
    """Calculate margin requirement for beta backing.

    Supports either a simple proxy ``sigma_ref × k`` or a broker-provided
    margin schedule with term-structure interpolation.
    """

    if financing_model == "schedule":
        ms = margin_schedule
        if ms is None:
            if schedule_path is None:
                raise ValueError("schedule_path required for schedule financing model")
            ms = load_margin_schedule(schedule_path)
        # Guard for static type checker
        if not isinstance(ms, pd.DataFrame):
            raise TypeError("margin_schedule must be a pandas DataFrame when provided")
        k = interpolate_margin_multiplier(term_months, ms)
    else:
        k = volatility_multiple

    return reference_sigma * k * total_capital


def validate_capital_allocation(
    external_pa_capital: float,
    active_ext_capital: float,
    internal_pa_capital: float,
    total_fund_capital: float = 1000.0,
    reference_sigma: float = 0.01,
    volatility_multiple: float = 3.0,
    *,
    financing_model: str = "simple_proxy",
    margin_schedule_path: Optional[Path] = None,
    term_months: float = 1.0,
) -> List[ValidationResult]:
    """Validate capital allocation including margin requirements.

    Args:
        external_pa_capital: External PA capital allocation
        active_ext_capital: Active extension capital allocation
        internal_pa_capital: Internal PA capital allocation
        total_fund_capital: Total fund capital available
        reference_sigma: Reference volatility for margin calculation
        volatility_multiple: Volatility multiple for margin calculation

    Returns:
        List of validation results
    """
    results = []

    # Calculate total allocated capital
    total_allocated = external_pa_capital + active_ext_capital + internal_pa_capital

    # Calculate margin requirement
    margin_requirement = calculate_margin_requirement(
        reference_sigma,
        volatility_multiple,
        total_fund_capital,
        financing_model=financing_model,
        schedule_path=margin_schedule_path,
        term_months=term_months,
    )

    # Check basic capital allocation
    if float(total_allocated) > float(total_fund_capital):
        results.append(
            ValidationResult(
                is_valid=False,
                message=f"Total allocated capital ({total_allocated:.1f}M) exceeds total fund capital ({total_fund_capital:.1f}M)",
                severity="error",
                details={
                    "total_allocated": total_allocated,
                    "total_available": total_fund_capital,
                    "excess": total_allocated - total_fund_capital,
                },
            )
        )

    # Check margin plus internal PA capital constraint
    margin_plus_internal = margin_requirement + internal_pa_capital
    if margin_plus_internal > total_fund_capital:
        results.append(
            ValidationResult(
                is_valid=False,
                message=f"Margin requirement ({margin_requirement:.1f}M) plus internal PA capital ({internal_pa_capital:.1f}M) "
                f"exceeds total capital ({total_fund_capital:.1f}M). "
                f"Consider reducing volatility multiple or internal PA allocation.",
                severity="error",
                details={
                    "margin_requirement": margin_requirement,
                    "internal_pa_capital": internal_pa_capital,
                    "total_requirement": margin_plus_internal,
                    "total_available": total_fund_capital,
                    "excess": margin_plus_internal - total_fund_capital,
                    "reference_sigma": reference_sigma,
                    "volatility_multiple": volatility_multiple,
                },
            )
        )

    # Provide buffer status information
    available_buffer = total_fund_capital - margin_plus_internal
    buffer_ratio = available_buffer / total_fund_capital if total_fund_capital > 0 else 0

    if float(buffer_ratio) < float(LOW_BUFFER_THRESHOLD):  # numeric comparison only
        severity = "warning" if buffer_ratio >= 0 else "error"
        results.append(
            ValidationResult(
                is_valid=buffer_ratio >= 0,
                message=f"Low capital buffer: {available_buffer:.1f}M ({buffer_ratio:.1%}) remaining after margin and internal PA",
                severity=severity,
                details={
                    "buffer_amount": available_buffer,
                    "buffer_ratio": buffer_ratio,
                    "margin_requirement": margin_requirement,
                },
            )
        )
    else:
        results.append(
            ValidationResult(
                is_valid=True,
                message=f"Capital buffer: {available_buffer:.1f}M ({buffer_ratio:.1%}) available",
                severity="info",
                details={
                    "buffer_amount": available_buffer,
                    "buffer_ratio": buffer_ratio,
                    "margin_requirement": margin_requirement,
                },
            )
        )

    return results


def validate_simulation_parameters(
    n_simulations: int, step_sizes: Dict[str, float] | None = None
) -> List[ValidationResult]:
    """Validate simulation parameters for extreme values.

    Args:
        n_simulations: Number of Monte Carlo simulations
        step_sizes: Dictionary of step sizes for parameter sweeps

    Returns:
        List of validation results
    """
    results = []

    # Check N_SIMULATIONS
    if n_simulations <= 10:
        results.append(
            ValidationResult(
                is_valid=True,
                message=(
                    f"N_SIMULATIONS={n_simulations} is very low and suitable only for testing"
                ),
                severity="warning",
                details={"value": n_simulations, "minimum": 10},
            )
        )
    elif n_simulations < 100:
        results.append(
            ValidationResult(
                is_valid=False,
                message=f"N_SIMULATIONS={n_simulations} is too low. Minimum recommended: 100",
                severity="error",
                details={"value": n_simulations, "minimum": 100},
            )
        )
    elif n_simulations < 1000:
        results.append(
            ValidationResult(
                is_valid=True,
                message=f"N_SIMULATIONS={n_simulations} is below recommended value of 1000 for stable results",
                severity="warning",
                details={"value": n_simulations, "recommended": 1000},
            )
        )
    elif n_simulations > 100000:
        results.append(
            ValidationResult(
                is_valid=True,
                message=f"N_SIMULATIONS={n_simulations} is very high and may cause long computation times",
                severity="warning",
                details={"value": n_simulations, "typical_max": 100000},
            )
        )

    # Check step sizes if provided
    if step_sizes:
        for param, step_size in step_sizes.items():
            if step_size <= 0:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        message=f"Step size for {param} must be positive, got {step_size}",
                        severity="error",
                        details={"parameter": param, "value": step_size},
                    )
                )
            elif (
                step_size < MIN_RECOMMENDED_STEP_SIZE
            ):  # Use named constant instead of magic number
                results.append(
                    ValidationResult(
                        is_valid=True,
                        message=f"Very small step size for {param} ({step_size}) may result in many parameter combinations",
                        severity="warning",
                        details={
                            "parameter": param,
                            "value": step_size,
                            "minimum_recommended": MIN_RECOMMENDED_STEP_SIZE,
                        },
                    )
                )

    return results


def format_validation_messages(
    results: List[ValidationResult], include_details: bool = False
) -> str:
    """Format validation results into a readable message.

    Args:
        results: List of validation results
        include_details: Whether to include detailed information

    Returns:
        Formatted message string
    """
    if not results:
        return "All validations passed."

    lines = []
    errors = [r for r in results if r.severity == "error"]
    warning_results = [r for r in results if r.severity == "warning"]
    infos = [r for r in results if r.severity == "info"]

    if errors:
        lines.append("❌ ERRORS:")
        for result in errors:
            lines.append(f"  • {result.message}")
            if include_details and result.details:
                lines.append(f"    Details: {result.details}")

    if warning_results:
        lines.append("⚠️ WARNINGS:")
        for result in warning_results:
            lines.append(f"  • {result.message}")
            if include_details and result.details:
                lines.append(f"    Details: {result.details}")

    if infos:
        lines.append("ℹ️ INFO:")
        for result in infos:
            lines.append(f"  • {result.message}")
            if include_details and result.details:
                lines.append(f"    Details: {result.details}")

    return "\n".join(lines)
