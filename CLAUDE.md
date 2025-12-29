# CLAUDE.md - Portable Alpha Extension Model

> **READ THIS FIRST** before making workflow changes.

## This is a Consumer Repo

This repository uses the **stranske/Workflows** workflow library. Most workflow logic lives there, not here.

**DO NOT** modify agent workflow files directly - they are synced from Workflows and will be overwritten.

## Architecture

```
stranske/Workflows (central library)
    │
    │ reusable workflows called via:
    │ uses: stranske/Workflows/.github/workflows/reusable-*.yml@main
    │
    ▼
This Repo (PAEM - consumer)
    .github/workflows/
      ├── agents-*.yml      → SYNCED from Workflows (don't edit)
      ├── autofix.yml       → SYNCED from Workflows (don't edit)
      ├── pr-00-gate.yml    → CUSTOM (can edit - PAEM-specific jobs)
      ├── ci.yml            → REPO-SPECIFIC (can edit)
      ├── release*.yml      → REPO-SPECIFIC (PAEM portable Windows zip)
      └── autofix-versions.env → REPO-SPECIFIC (can edit)
```

## Which Files Can Be Edited

| File | Editable? | Notes |
|------|-----------|-------|
| `agents-*.yml` | ❌ No | Synced from Workflows - changes will be overwritten |
| `autofix.yml` | ❌ No | Synced from Workflows |
| `pr-00-gate.yml` | ✅ Yes | Custom Gate with PAEM-specific jobs |
| `ci.yml` | ✅ Yes | PAEM-specific CI configuration |
| `release*.yml` | ✅ Yes | PAEM portable Windows zip packaging |
| `autofix-versions.env` | ✅ Yes | Version pins for autofix tools |

## Common Issues

### Workflow fails with "workflow file issue"
- A reusable workflow is being called that doesn't exist
- Check Workflows repo has the required `reusable-*.yml` file
- Consumer workflows call INTO Workflows repo, not local files

### Agent not picking up changes
- Check PR has `agent:codex` label
- Check Gate workflow passed (green checkmark)
- Check PR body has unchecked tasks

### Need to update agent workflows
- DON'T edit locally - changes will be overwritten
- Fix in Workflows repo → sync will propagate here
- Or request manual sync: `gh workflow run maint-68-sync-consumer-repos.yml --repo stranske/Workflows`

## Reference Implementation

**Travel-Plan-Permission** is the reference consumer repo. When debugging:
1. Check if it works there first
2. Compare this repo's `.github/` with Travel-Plan-Permission
3. Look for missing files or differences

## Workflows Documentation

For detailed docs, see **stranske/Workflows**:
- `docs/INTEGRATION_GUIDE.md` - How consumer repos work
- `docs/keepalive/GoalsAndPlumbing.md` - Keepalive system design
- `docs/keepalive/SETUP_CHECKLIST.md` - Required files and secrets

---

## 🔀 POLICY: Cross-Repo Work

> **CRITICAL**: Read this before ANY work that might affect the Workflows repo.

### Policy Checkpoint Trigger

When creating a todo list, ask:

**"Does this work need changes in stranske/Workflows?"**

Signs that you need Workflows changes:
- Adding a new agent capability
- Modifying how keepalive/autofix/verifier works
- Needing a new Codex prompt
- Bug in a reusable workflow

### If YES → Work in Workflows First

1. Clone/checkout stranske/Workflows
2. Make changes there (following Workflows CLAUDE.md policy)
3. Ensure sync manifest is updated
4. Trigger sync to propagate to this repo
5. Then verify in this repo

**DO NOT** try to fix Workflows issues by editing local files - they will be overwritten on next sync.

### Quick Commands

```bash
# Check if a file is synced (compare to template)
diff .github/workflows/autofix.yml \
     <(gh api repos/stranske/Workflows/contents/templates/consumer-repo/.github/workflows/autofix.yml --jq '.content' | base64 -d)

# Trigger sync from Workflows
gh workflow run maint-68-sync-consumer-repos.yml --repo stranske/Workflows -f repos="stranske/Portable-Alpha-Extension-Model"

# Check sync manifest for what SHOULD be here
gh api repos/stranske/Workflows/contents/.github/sync-manifest.yml --jq '.content' | base64 -d
```

---

## PAEM-Specific Development

### Bootstrap and Setup
```bash
# Initial setup using dev script (RECOMMENDED)
./dev.sh setup

# Alternative: Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**TIMING**: Setup takes 3-5 minutes initially. NEVER CANCEL during dependency installation.

### Build and Test Commands
```bash
# Full CI pipeline - takes 15 seconds. NEVER CANCEL.
./dev.sh ci

# Individual checks
./dev.sh lint            # Ruff linting - takes <1 second
./dev.sh typecheck       # Pyright type checking - takes 10 seconds  
./dev.sh test            # Pytest test suite - takes 2-3 seconds
```

### Run the Application
```bash
# Basic simulation (takes 16 seconds - NEVER CANCEL)
python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml \
  --index data/sp500tr_fred_divyield.csv --output Results.xlsx

# Parameter sweep modes (takes 4-16 seconds - NEVER CANCEL)
python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml \
  --index data/sp500tr_fred_divyield.csv --mode returns --output ReturnsSweep.xlsx

# Launch Streamlit dashboard (starts in 10-15 seconds)
./dev.sh dashboard
```

### File Locations
```
├── pa_core/                 # Main Python package
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration handling  
│   ├── agents/             # Simulation agents
│   ├── data/               # Data loading/conversion
│   ├── reporting/          # Excel/export functionality
│   └── viz/                # Visualization components
├── dashboard/              # Streamlit dashboard
├── tests/                  # Test suite (pytest)
├── config/                 # Configuration templates
├── templates/              # YAML scenario templates  
└── data/                   # Sample index data
```

### Troubleshooting

**Import Errors**:
```bash
source .venv/bin/activate
export PYTHONPATH=/workspaces/Portable-Alpha-Extension-Model
```

**Dashboard Not Starting**:
```bash
pip list | grep streamlit
python -m streamlit run dashboard/app.py --server.headless=true --server.port=8501
# Allow 10-15 seconds for startup - NEVER CANCEL
```

**NEVER CANCEL REMINDER**: All build, test, and simulation commands complete in under 30 seconds. Set appropriate timeouts and wait for completion.
