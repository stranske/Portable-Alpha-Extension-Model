# CRITICAL DISCOVERY: Analysis Mode Functionality Gap

## Issue Summary
The program documentation and parameter files suggest 4 analysis modes (`capital`, `returns`, `alpha_shares`, `vol_mult`), but the current CLI (`pa_core.cli`) **does not implement parameter sweep functionality**.

## Evidence:
1. **ModelConfig has no `analysis_mode` field** - The configuration schema doesn't recognize this parameter
2. **CLI only runs single simulations** - No sweep logic in current codebase  
3. **Parameter sweep code is archived** - Found in `archive/Old/Portable_Alpha_Visualizations.py`
4. **Working parameters.csv uses sweep format** - But CLI ignores sweep-specific parameters

## Current Status:
- ✅ **Single-point simulation**: Works via CLI with simple config files
- ❌ **Parameter sweeps**: Legacy functionality, not implemented in current CLI
- ❌ **4 analysis modes**: Documentation mentions them but they don't work

## Impact on Users:
- **Misleading documentation**: Tutorials suggest 4 modes but only 1 works
- **Confusing parameter files**: Templates don't match actual CLI capabilities  
- **Analysis mode parameter**: Required but ignored by the system

## Recommended Fix:
1. Update documentation to clarify current CLI only supports single simulations
2. Create proper single-simulation templates for each "conceptual mode"
3. Either implement parameter sweep functionality or remove references to it
4. Make analysis_mode optional or remove it from requirements

## For Tutorial 1:
Users should use simple parameter files focused on single simulations, not parameter sweeps.
