# Development Workflow Guide

## Work Division Strategy

### Codex (Core Function Improvements)
- **Parameter sweep engine** - Implementation of 4 analysis modes
- **New simulation features** - Advanced Monte Carlo algorithms
- **Major architectural changes** - Core system improvements
- **Performance optimizations** - Speed and memory improvements
- **New agent types** - Additional strategy implementations

### Human Assistant (Debugging, Formatting, Linting)
- **Bug fixes** - Error handling and edge cases
- **Code formatting** - Black, isort, consistent style
- **Type hints** - MyPy improvements and annotations
- **Test improvements** - Coverage and test quality
- **Configuration validation** - Input validation and error messages
- **User experience polish** - Documentation and usability

## Daily Workflow

### 1. Check for Updates
```bash
make check-updates
```
This checks if Codex has pushed new changes to main.

### 2. Sync with Main (if updates available)
```bash
make sync
```
This pulls the latest changes from the remote main branch.

### 3. Work on Your Branch
```bash
# Create a new branch for your work
git checkout -b polish/fix-validation-errors

# Make your changes...
# Run development checks
make dev-check
```

### 4. Push Your Changes
```bash
git add .
git commit -m "fix: improve parameter validation error messages"
git push origin polish/fix-validation-errors
```

### 5. Merge Back to Main
```bash
git checkout main
git merge polish/fix-validation-errors
git push origin main
```

## Development Commands

### Code Quality
```bash
make format      # Format code with black/isort
make lint        # Check with flake8/ruff
make typecheck   # Check types with mypy
make test        # Run tests with coverage
make dev-check   # Run all checks
```

### Git Workflow
```bash
make sync           # Pull latest main
make check-updates  # Check for remote updates
make sync-rebase    # Rebase current branch on main
```

### Running the System
```bash
make demo       # Run CLI with sample data
make dashboard  # Start Streamlit dashboard
```

## Codespace Auto-Sync

The codespace automatically syncs with GitHub. When Codex pushes changes:

1. **Check for updates**: `make check-updates`
2. **Pull updates**: `make sync`
3. **Rebase your work**: `make sync-rebase` (if you have uncommitted changes)

## Branch Naming Convention

- **Codex branches**: `feature/parameter-sweep-engine`, `feature/new-agents`
- **Polish branches**: `polish/formatting`, `polish/bug-fixes`, `polish/validation`
- **Hotfix branches**: `hotfix/critical-bug-fix`

## Merge Strategy

1. **Feature work** (Codex) → Direct to main after testing
2. **Polish work** (Human) → Branch → Review → Merge to main
3. **Hotfixes** → Direct to main → Inform other party

## Avoiding Conflicts

### Codex Focus Areas:
- `pa_core/simulations.py`
- `pa_core/agents/` (new files)
- New modules and features

### Human Focus Areas:
- `pa_core/cli.py` (error handling)
- `dashboard/app.py` (UX improvements)
- Test files
- Configuration files
- Documentation

## Communication

When major changes are made:
1. **Update this file** with new workflow notes
2. **Document breaking changes** in commit messages
3. **Tag releases** for major milestones

## Emergency Sync

If branches diverge significantly:
```bash
# Reset to remote main (CAUTION: loses local changes)
git fetch origin
git reset --hard origin/main

# Or create a backup first
git branch backup-my-work
git reset --hard origin/main
```
