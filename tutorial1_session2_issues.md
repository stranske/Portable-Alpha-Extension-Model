# Tutorial 1 User Experience Issues - Session 2

## Critical Usability Problems for New Users

### 1. **Missing Critical Parameter Documentation**
**Issue**: The tutorial doesn't mention the mandatory "Analysis mode" parameter.
**Impact**: User following tutorial exactly will get CLI failure with no clear explanation.
**Fix Needed**: Tutorial should explicitly mention this parameter and explain the modes:
- `returns`: Fixed returns/volatilities (recommended for beginners)
- `capital`: Capital allocation sweeps
- `alpha_shares`: Alpha allocation sweeps  
- `vol_mult`: Volatility multiplier sweeps

### 2. **Template Mismatch Confusion**
**Issue**: Templates in `config/` don't match the working `parameters.csv` format.
**Impact**: New users get confused about which template to use.
**User Expectation**: Templates should "just work" when copied.
**Fix Needed**: Templates should include all mandatory parameters including "Analysis mode".

### 3. **Extreme Results Indicating Parameter Issues**
**Issue**: ActiveExtension agent shows 5,669,701% return - clearly unrealistic.
**Impact**: New users would think the software is broken.
**Root Cause**: Likely unrealistic parameter combinations in my configuration.
**Fix Needed**: Parameter validation to warn about unrealistic combinations.

### 4. **Parameter Business Logic Not Explained**
**Issue**: Tutorial assumes users know what "Active share (%)" means in this context.
**Impact**: Portable alpha experts may not understand script-specific notation.
**Fix Needed**: Parameter guide explaining business meaning of each parameter.

### 5. **No Guidance on Realistic Parameter Ranges**
**Issue**: No indication of what constitutes realistic vs unrealistic parameter values.
**Impact**: Users may enter nonsensical combinations.
**Fix Needed**: Parameter validation with realistic ranges and warnings.

## Successful Aspects

✅ **CLI Command Structure**: The basic command worked as documented
✅ **Output Generation**: Excel file was created successfully  
✅ **Console Summary**: Rich table provides clear metric explanations
✅ **Error Recovery**: System didn't crash despite unrealistic parameters

## Next Steps for Session 2

1. Fix the parameter validation issue causing extreme returns
2. Update Tutorial 1 to include "Analysis mode" parameter
3. Create proper beginner-friendly templates
4. Add parameter validation warnings
5. Continue to Tutorial 2 with corrected parameters
