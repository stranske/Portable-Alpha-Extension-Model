# Portable Alpha Model - User Testing Issues and Improvements

## Issues Identified

### 1. **Critical: Missing ShortfallProb Calculation**
- The configuration mandates ShortfallProb in risk_metrics 
- But summary_table() doesn't calculate it
- Visualization code expects it but gets None/0.0

### 2. **Missing Breach Probability Defaults**
- BreachProb shows as None because no breach_threshold provided
- No guidance on what threshold to use
- User guide mentions this metric but it's not calculated

### 3. **Poor User Experience for New Users**
- No explanation of what the metrics mean
- Missing context for interpreting the numbers  
- No guidance on typical ranges or thresholds

### 4. **Configuration File Issues**
- parameters.csv was missing risk_metrics (fixed)
- No clear documentation of required vs optional parameters
- Template differences between CSV and YAML formats

## Proposed Improvements

### 1. Enhanced Summary Table Function
- Add ShortfallProb calculation with configurable threshold
- Provide sensible defaults for breach_threshold
- Add metric descriptions for new users

### 2. Better CLI Output
- Add explanatory text for metrics
- Show thresholds used for calculations
- Provide interpretation guidance

### 3. Improved Configuration Templates
- Ensure both CSV and YAML have all required fields
- Add better documentation and examples
- Include commonly used thresholds

### 4. Enhanced User Guide Testing
- Step-by-step walkthrough with expected outputs
- Common troubleshooting scenarios
- Clear explanations of all metrics

## Testing Notes

The basic CLI works but lacks the promised ShortfallProb metric. This would significantly confuse new users following the tutorial.
