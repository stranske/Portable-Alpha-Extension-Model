# 🤖 Codex Automation Quick Start

## After Reopening Codespace

### 1. Environment Setup (First Time)
```bash
# Set up development environment
./setup.sh
# OR
make install-dev
```

### GitHub Token Permissions

Automation workflows use the GitHub CLI and require authentication. Provide a personal access token via the `CODEX_PAT` secret or rely on the default `GITHUB_TOKEN` with these permissions:

- contents: write
- pull-requests: write
- issues: write
- actions: read
- checks: write

### 2. Run Automation (Any Time)
```bash
# Quick debugging check
make debug-codex

# Complete validation (recommended)
make validate-pr

# Auto-fix and commit
make debug-codex-fix
```

### 3. Direct Script Usage
```bash
# Basic debugging
python scripts/debug_codex_pr.py --branch=$(git branch --show-current)

# With auto-commit
python scripts/debug_codex_pr.py --branch=$(git branch --show-current) --commit

# Generate detailed report
python scripts/debug_codex_pr.py --branch=$(git branch --show-current) --report=debug.md
```

### 4. Key Files Preserved
- ✅ `scripts/debug_codex_pr.py` - Main automation script
- ✅ `.github/workflows/codex-auto-debug.yml` - CI/CD automation
- ✅ `docs/AUTOMATED_DEBUGGING.md` - Complete documentation
- ✅ `.vscode/tasks.json` - VS Code task integration
- ✅ `Makefile` - Enhanced with automation targets

### 5. What It Does
- 🔍 **Detects**: Type errors, import issues, formatting violations
- 🔧 **Fixes**: Code formatting, import cleanup, common patterns
- ✅ **Validates**: Tests, type checking, linting
- 📊 **Reports**: Comprehensive debugging analysis
- 🚀 **Integrates**: GitHub Actions, VS Code, command line

### 6. Workflow
1. **Automatic Detection**: System detects Codex branch changes
2. **Issue Analysis**: Scans for common integration problems
3. **Auto-fixing**: Applies standard fixes automatically
4. **Manual Guidance**: Reports complex issues for review
5. **Validation**: Ensures all tests pass and code quality maintained

The automation system is **production-ready** and scales your Codex collaboration workflow!
