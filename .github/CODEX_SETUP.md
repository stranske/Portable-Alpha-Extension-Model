# 🤖 Codex Auto-Debug Setup Guide

## GitHub Actions Permissions Fix

The Codex auto-debug workflow requires proper GitHub permissions to:
- Comment on Pull Requests
- Push auto-fix commits
- Access workflow artifacts

## ✅ **Current Fix Applied**

The workflow now includes explicit permissions and improved error handling:

```yaml
permissions:
  contents: write          # Push commits
  pull-requests: write     # Comment on PRs
  issues: write           # PR comments (PRs are issues)
  actions: read           # Access artifacts
```

## 🔧 **If Issues Persist (Fork PRs)**

For Pull Requests from **forks**, you may need a Personal Access Token (PAT):

### 1. Create a Personal Access Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token" → "Generate new token (classic)"
3. Name: `Codex Auto-Debug PAT`
4. Select scopes:
   - ✅ `repo` (Full repository access)
   - ✅ `workflow` (Update GitHub Action workflows)
   - ✅ `write:discussion` (Create and update PR comments)

### 2. Add PAT as Repository Secret

1. Go to your repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `CODEX_PAT`
4. Value: Paste your PAT token
5. Click "Add secret"

### 3. Workflow Auto-Configuration

The workflow automatically uses the PAT if available:

```yaml
token: ${{ secrets.CODEX_PAT || secrets.GITHUB_TOKEN }}
```

## 🚀 **Testing the Fix**

1. Create a new Codex branch: `git checkout -b codex/test-permissions`
2. Make a small change and push
3. Open a Pull Request to `main`
4. The workflow should run and comment on the PR

## 🔍 **Troubleshooting**

### Error: "Resource not accessible by integration"
- ✅ **Fixed**: Added explicit `permissions` block
- ✅ **Enhanced**: PAT fallback for fork PRs
- ✅ **Improved**: Better error handling and logging

### Error: "Permission denied" on git push
- ✅ **Fixed**: Added `contents: write` permission
- ✅ **Enhanced**: Better error messages in push step

### Error: Cannot comment on PR
- ✅ **Fixed**: Added `pull-requests: write` and `issues: write`
- ✅ **Enhanced**: PAT option for enhanced permissions

## 📊 **Expected Workflow Behavior**

1. **Trigger**: PR opened/updated on `codex/*` branch
2. **Setup**: Install Python dependencies
3. **Debug**: Run `scripts/debug_codex_pr.py`
4. **Report**: Upload debug report as artifact
5. **Comment**: Post results to PR (with proper permissions)
6. **Auto-fix**: Commit fixes if found (with proper permissions)

## ✅ **Verification**

The workflow now handles permissions correctly and should resolve the "Resource not accessible by integration" error.
