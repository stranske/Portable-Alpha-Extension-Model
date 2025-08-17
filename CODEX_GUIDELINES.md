# Codex Development Guidelines

## 🚨 CRITICAL: READ BEFORE CODING

### Current Repository State (July 12, 2025)

**Repository Status:** ✅ STABLE - Core functionality working correctly

**Recently Fixed Issues:**
- ✅ ActiveExtensionAgent percentage conversion bug - **WORKING, DO NOT MODIFY**
- ✅ CLI validation and error handling
- ✅ Dashboard configuration handling
- ✅ All tests passing (except test config issues being resolved)

## 🎯 YOUR FOCUS AREAS

### 1. Parameter Sweep Engine (TOP PRIORITY)
**Goal:** Implement 4 analysis modes for systematic parameter exploration

**Location:** `pa_core/cli.py` and `pa_core/simulations.py`

**Current Gap:** CLI only runs single simulations, but users expect parameter sweeps:
- `capital` mode: Vary capital allocations across sleeves
- `returns` mode: Vary expected returns and volatilities  
- `alpha_shares` mode: Vary alpha/beta share splits
- `vol_mult` mode: Stress test with volatility multipliers

**Implementation Needed:**
```python
# Add to cli.py
def run_parameter_sweep(config: ModelConfig, mode: str) -> pd.DataFrame:
    """Run systematic parameter variations based on mode"""
    pass

# Usage should be:
# python -m pa_core.cli --config config.yml --mode capital --sweep-range 0.1,0.5,0.1
```

**Reference:** See `CODEX_IMPLEMENTATION_SPEC.md` for complete requirements

### 2. New Agent Types (SECONDARY)
**Goal:** Add new strategy implementations

**Pattern to Follow:**
```python
class NewStrategyAgent(Agent):
    def monthly_returns(self, r_beta: Array, alpha_stream: Array, financing: Array) -> Array:
        # Your implementation
        pass
```

**Requirements:**
- Inherit from `Agent` base class
- Implement `monthly_returns` method
- Add to `pa_core/agents/registry.py`
- Add tests in `tests/test_agents.py`

### 3. Performance Improvements (TERTIARY)
**Goal:** Optimize simulation speed and memory usage

**Focus Areas:**
- Vectorization improvements in `pa_core/simulations.py`
- Memory-efficient array operations
- Parallel processing for large parameter sweeps

## ❌ DO NOT MODIFY

### Working Code (Already Fixed)
- `pa_core/agents/active_ext.py` - Percentage conversion working correctly
- `pa_core/cli.py` - Basic CLI functionality working
- `dashboard/app.py` - Core dashboard working
- Test files - Core tests are passing

### Parallel Development (Human Assistant Handling)
- Code formatting and style
- Linting and type hints
- Documentation polishing
- Minor bug fixes and error messages

## 🛠️ Development Workflow

### Before You Start
1. **Check for updates:** `make check-updates`
2. **Pull latest:** `make sync` (if updates available)
3. **Create feature branch:** `git checkout -b feature/your-feature-name`

### While Developing
1. **Run tests frequently:** `python -m pytest tests/test_[relevant].py -v`
2. **Test your changes:** Use `my_first_scenario.yml` for validation
3. **Follow existing patterns:** Look at current agent implementations

### Before Pushing
1. **Run full test suite:** `python -m pytest tests/ -v`
2. **Test with sample data:** `python -m pa_core.cli --config my_first_scenario.yml`
3. **Commit with clear messages:** `git commit -m "feat: add parameter sweep engine"`
4. **Push to feature branch:** `git push origin feature/your-feature-name`

## 📋 Success Criteria

### Parameter Sweep Engine
- [ ] Users can specify `--mode capital` to vary capital allocations
- [ ] Users can specify `--mode returns` to vary return assumptions
- [ ] Users can specify `--mode alpha_shares` to vary alpha/beta splits  
- [ ] Users can specify `--mode vol_mult` to stress test volatilities
- [ ] Output includes all parameter combinations and their results
- [ ] Backward compatibility maintained (single runs still work)

### New Agents
- [ ] New agent class follows established patterns
- [ ] Tests added and passing
- [ ] Registry updated for auto-discovery
- [ ] Documentation updated

### Performance
- [ ] Measurable speed improvements on large simulations
- [ ] Memory usage optimized for parameter sweeps
- [ ] No regression in accuracy or functionality

## 🤝 Communication

**If you're unsure about anything:**
1. Check this file first
2. Look at `CODEX_IMPLEMENTATION_SPEC.md` for detailed specs
3. Examine existing working code patterns
4. Run tests to validate your understanding

**If tests fail:**
1. Check if you modified working code that shouldn't be changed
2. Verify your changes follow existing patterns
3. Test with the provided sample configurations

**Remember:** The goal is to ADD functionality, not fix what's already working!
