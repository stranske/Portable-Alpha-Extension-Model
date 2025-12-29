# PAEM → Workflows Consumer Repo Transition Plan

**Document Created**: December 29, 2025  
**Status**: Ready to Execute  
**Approach**: Feature Branch (Option 4)  

## Overview

Transition Portable-Alpha-Extension-Model (PAEM) from its standalone workflow system to become a consumer repo of `stranske/Workflows`.

### Key Decisions

| Decision | Choice |
|----------|--------|
| Multi-platform matrix (Windows/macOS) | **DROP** - Linux only |
| Deprecated workflows | **ARCHIVE** with 3-month deletion reminder |
| PAEM issues | **CONVERT** to keepalive format |
| CI-related issues | Mark as **"not planned"** |
| Coverage target | **85%** (create tracking issue) |
| Custom Gate | **YES** - PAEM has domain-specific jobs |
| Codespace validation | Retain and offer to other repos |

### Resources Available

| Resource | Location/Access |
|----------|-----------------|
| Cross-repo PAT | Environment: `CODESPACES` |
| Consumer Sync Skill | `stranske/Workflows/.github/copilot-skills-local/consumer-sync-process.md` |
| Reference consumer | `stranske/Travel-Plan-Permission` |
| Secondary reference | `stranske/Manager-Database` (has custom ci.yml) |

---

## Phase 1: Preparation (Before Starting Work)

### 1.1 Create Feature Branch

```bash
git checkout -b feature/workflows-transition
git push -u origin feature/workflows-transition
```

### 1.2 Disable Branch Protection Temporarily (if needed)

For the transition PR, you may need to bypass branch protection:
- Go to **Settings → Branches → main**
- Temporarily allow the transition PR to merge without Gate (or use bypass)

### 1.3 Verify Secrets

Required secrets for consumer repo operation:

| Secret | Purpose | Check |
|--------|---------|-------|
| `SERVICE_BOT_PAT` | Bot account for comments/labels | Verify exists in repo settings |
| `ACTIONS_BOT_PAT` | Workflow dispatch triggers (optional) | Verify or use SERVICE_BOT_PAT |
| `OWNER_PR_PAT` | PR creation on behalf of owner (optional) | Verify or use SERVICE_BOT_PAT |

```bash
# Verify secrets are configured
gh secret list --repo stranske/Portable-Alpha-Extension-Model
```

---

## Phase 2: Disable Old CI (Immediate)

### 2.1 Add `if: false` to Deprecated Workflows

Add `if: false` at the job level to prevent execution while preserving files:

**Files to disable** (in `.github/workflows/`):
- `ci.yml` - Will be replaced with consumer `ci.yml`
- `autofix.yml` - Will be replaced with synced `autofix.yml`
- `codex-route-and-kickoff.yml` - Replaced by agent workflows
- `codex-auto-debug.yml` - Replaced by agent workflows
- `streamlined-codex-debug.yml` - Replaced by agent workflows
- `debug-on-failure.yml` - Replaced by agent workflows
- `agent-route-and-kickoff.yml` - Replaced by agent workflows
- `assign-to-copilot.yml` - Replaced by agent workflows
- `label-agent-prs.yml` - Replaced by `agents-pr-meta.yml`
- `label-to-assignee.yml` - Replaced by agent workflows
- `smoke.yml` - Merged into new ci.yml
- `risk-bucket.yml` - Review for keepalive conversion
- `enable-automerge.yml` - Replaced by agent workflows
- `issue-event-tracer.yml` - Replaced by agent workflows
- `sync-labels.yml` - Keep or replace

**Files to RETAIN as-is**:
- `release-packaging.yml` - PAEM-specific portable Windows zip
- `release.yml` - May need minor updates for new Gate

Example disable pattern:
```yaml
jobs:
  ci:
    if: false  # TRANSITION: Disabled pending Workflows migration - delete after 2025-03-29
    runs-on: ubuntu-latest
    # ... rest of job
```

---

## Phase 3: Add Consumer Repo Files

### 3.1 Copy Thin Caller Workflows

From `stranske/Workflows/templates/consumer-repo/.github/workflows/`:

```bash
# Fetch consumer workflow templates
WORKFLOWS_RAW="https://raw.githubusercontent.com/stranske/Workflows/main/templates/consumer-repo/.github/workflows"

curl -sL "$WORKFLOWS_RAW/ci.yml" -o .github/workflows/ci-new.yml
curl -sL "$WORKFLOWS_RAW/pr-00-gate.yml" -o .github/workflows/pr-00-gate.yml
curl -sL "$WORKFLOWS_RAW/autofix.yml" -o .github/workflows/autofix-new.yml
curl -sL "$WORKFLOWS_RAW/agents-issue-intake.yml" -o .github/workflows/agents-63-issue-intake.yml
curl -sL "$WORKFLOWS_RAW/agents-orchestrator.yml" -o .github/workflows/agents-70-orchestrator.yml
curl -sL "$WORKFLOWS_RAW/agents-pr-meta.yml" -o .github/workflows/agents-pr-meta.yml
curl -sL "$WORKFLOWS_RAW/agents-verifier.yml" -o .github/workflows/agents-verifier.yml
curl -sL "$WORKFLOWS_RAW/autofix-versions.env" -o .github/workflows/autofix-versions.env
```

### 3.2 Customize ci.yml for PAEM

Replace `YOUR_PACKAGE_NAME` → `pa_core`

Key customizations needed:
- Test paths: `tests/`
- Source paths: `pa_core/`
- Python version: `3.11` (or `3.12`, `3.13`)
- Coverage threshold: `85`
- Retain Codespace validation job (domain-specific)

### 3.3 Create Custom Gate Workflow

PAEM needs a custom Gate because of domain-specific jobs:
- Codespace validation
- Integration tests (golden tutorials)

Reference implementations:
- `stranske/Travel-Plan-Permission/.github/workflows/pr-00-gate.yml`
- `stranske/Manager-Database/.github/workflows/pr-00-gate.yml`

Template for PAEM custom Gate:
```yaml
name: Gate

on:
  pull_request:
    branches: [main]
  workflow_run:
    workflows: ["CI", "Autofix"]
    types: [completed]
    branches-ignore: [main]

jobs:
  python-ci:
    if: github.event_name == 'pull_request'
    uses: stranske/Workflows/.github/workflows/reusable-10-ci-python.yml@main
    with:
      python_versions: '["3.11"]'  # PAEM: Single version, no multi-platform
      primary_version: "3.11"
      min_coverage: 85
      fail_under_coverage: true
      ruff_check: true
      mypy_check: true
      module_name: pa_core
    secrets:
      token: ${{ secrets.SERVICE_BOT_PAT || secrets.GITHUB_TOKEN }}

  codespace-validation:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate Codespace Config
        run: |
          bash -n .devcontainer/setup.sh
          echo "✅ Codespace setup script syntax is valid"

  gate-summary:
    needs: [python-ci, codespace-validation]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Check Gate Status
        run: |
          if [[ "${{ needs.python-ci.result }}" != "success" ]] || \
             [[ "${{ needs.codespace-validation.result }}" != "success" ]]; then
            echo "❌ Gate failed"
            exit 1
          fi
          echo "✅ Gate passed"
```

### 3.4 Create CLAUDE.md

Adapt `.github/copilot-instructions.md` content into `CLAUDE.md` format:

```bash
# Create CLAUDE.md in repo root
cp .github/copilot-instructions.md CLAUDE.md
```

Then add consumer repo header (from `stranske/Workflows/templates/consumer-repo/CLAUDE.md`):
- Architecture section
- Cross-repo policy section
- Common issues section
- Reference implementation links

---

## Phase 4: Archive Deprecated Workflows

### 4.1 Move to Archive Folder

```bash
mkdir -p archive/workflows-deprecated-2025-12

# Move deprecated workflows
for f in ci autofix codex-route-and-kickoff codex-auto-debug \
         streamlined-codex-debug debug-on-failure agent-route-and-kickoff \
         assign-to-copilot label-agent-prs label-to-assignee smoke \
         risk-bucket enable-automerge issue-event-tracer sync-labels; do
  if [ -f ".github/workflows/${f}.yml" ]; then
    mv ".github/workflows/${f}.yml" "archive/workflows-deprecated-2025-12/"
  fi
done
```

### 4.2 Create Deletion Reminder Issue

Create issue with label `reminder`:

**Title**: `🗑️ Delete archived workflows after 2025-03-29`

**Body**:
```markdown
## Reminder: Delete Deprecated Workflows

The following workflows were archived during the Workflows consumer transition on 2025-12-29.
They should be deleted after 3 months (2025-03-29) if no issues arise.

**Location**: `archive/workflows-deprecated-2025-12/`

**Files**:
- ci.yml
- autofix.yml
- codex-route-and-kickoff.yml
- codex-auto-debug.yml
- streamlined-codex-debug.yml
- debug-on-failure.yml
- agent-route-and-kickoff.yml
- assign-to-copilot.yml
- label-agent-prs.yml
- label-to-assignee.yml
- smoke.yml
- risk-bucket.yml
- enable-automerge.yml
- issue-event-tracer.yml
- sync-labels.yml

## Acceptance Criteria
- [ ] No issues reported with new Workflows system
- [ ] All agent workflows functioning correctly
- [ ] Delete the `archive/workflows-deprecated-2025-12/` folder
- [ ] Close this issue
```

---

## Phase 5: Label and Issue Management

### 5.1 Verify Required Labels

Ensure these labels exist (from Workflows label system):

```bash
# Core labels
gh label create "agent:codex" --color "0E8A16" --description "Assign to Codex agent" --force
gh label create "agent:copilot" --color "1D76DB" --description "Assign to Copilot" --force
gh label create "agents" --color "FBCA04" --description "Agent automation" --force
gh label create "agents:pause" --color "B60205" --description "Pause agent work" --force
gh label create "autofix:applied" --color "7057FF" --description "Autofix was applied" --force

# Status labels
gh label create "reminder" --color "D4C5F9" --description "Reminder for future action" --force
```

### 5.2 Convert PAEM Issues to Keepalive Format

For existing feature issues, add keepalive structure:

```markdown
## Why
[Context and rationale]

## Tasks
- [ ] Task 1
- [ ] Task 2

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
```

### 5.3 Mark CI-Related Issues as "not planned"

Issues related to the old CI system should be closed:
- Reference this transition in close comment
- Use "not planned" close reason

### 5.4 Create Coverage Threshold Issue

**Title**: `📊 Increase test coverage to 85%`

**Labels**: `agent:codex`, `testing`, `enhancement`

**Body**:
```markdown
## Why
Establish 85% code coverage as baseline for PAEM as part of Workflows consumer transition.

## Tasks
- [ ] Audit current coverage and identify gaps
- [ ] Add tests for uncovered `pa_core` modules
- [ ] Add tests for `dashboard` modules
- [ ] Update CI to enforce 85% minimum

## Acceptance Criteria
- [ ] Coverage >= 85% on main branch
- [ ] CI fails if coverage drops below 85%
- [ ] Coverage badge shows current percentage
```

---

## Phase 6: Validation and Merge

### 6.1 Pre-Merge Checklist

- [ ] All new workflows have valid YAML syntax
- [ ] `pr-00-gate.yml` calls correct reusable workflow
- [ ] `ci.yml` has PAEM-specific customizations
- [ ] `CLAUDE.md` created with consumer repo context
- [ ] Deprecated workflows archived (not deleted)
- [ ] 3-month deletion reminder issue created
- [ ] Required labels exist
- [ ] Secrets verified

### 6.2 Test on Feature Branch

```bash
# Push changes and wait for CI
git add .
git commit -m "feat: transition to Workflows consumer repo

- Add thin caller workflows from stranske/Workflows
- Create custom Gate with Codespace validation
- Archive deprecated workflows (3-month retention)
- Add CLAUDE.md with consumer repo context
- Configure for single Python version (Linux only)"

git push
```

### 6.3 Verify Gate Workflow

- Check that `pr-00-gate.yml` triggers
- Verify it calls `stranske/Workflows/.github/workflows/reusable-10-ci-python.yml@main`
- Confirm all jobs pass

### 6.4 Merge Strategy

1. Get PR reviewed (or bypass for transition)
2. Merge using **Squash and merge**
3. Verify main branch CI passes
4. Trigger manual sync from Workflows to ensure alignment

```bash
# After merge, trigger sync to confirm integration
gh workflow run maint-68-sync-consumer-repos.yml \
  --repo stranske/Workflows \
  -f repos="stranske/Portable-Alpha-Extension-Model" \
  -f dry_run=true
```

---

## Post-Transition Tasks

### Register PAEM as Consumer Repo

Request addition to Workflows sync manifest:
1. Open issue in `stranske/Workflows`
2. Request PAEM be added to `REGISTERED_CONSUMER_REPOS`
3. This enables automatic template syncing

### Offer Codespace Validation to Other Repos

PAEM's Codespace validation job could benefit other consumer repos:
- Document the pattern
- Consider proposing as reusable workflow or template addition

### Monitor First Week

- Watch for agent workflow triggering
- Verify autofix works correctly
- Check keepalive loop functionality (if issues with `agent:codex` label exist)

---

## Rollback Plan

If transition fails:

1. **Restore old workflows from archive**:
   ```bash
   mv archive/workflows-deprecated-2025-12/*.yml .github/workflows/
   ```

2. **Remove new consumer workflows**:
   ```bash
   rm .github/workflows/pr-00-gate.yml
   rm .github/workflows/agents-*.yml
   # Restore original ci.yml and autofix.yml
   ```

3. **Revert the merge commit**:
   ```bash
   git revert <MERGE_COMMIT_SHA>
   ```

---

## Reference Links

- [Workflows SETUP_CHECKLIST](https://github.com/stranske/Workflows/blob/main/docs/keepalive/SETUP_CHECKLIST.md)
- [Workflows INTEGRATION_GUIDE](https://github.com/stranske/Workflows/blob/main/docs/INTEGRATION_GUIDE.md)
- [Consumer Repo CLAUDE.md Template](https://github.com/stranske/Workflows/blob/main/templates/consumer-repo/CLAUDE.md)
- [Travel-Plan-Permission (Reference)](https://github.com/stranske/Travel-Plan-Permission)
- [Manager-Database (Custom CI Reference)](https://github.com/stranske/Manager-Database)
- [Consumer Sync Process Skill](https://github.com/stranske/Workflows/blob/main/.github/copilot-skills-local/consumer-sync-process.md)
