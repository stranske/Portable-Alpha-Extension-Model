#!/usr/bin/env python3
"""
Streamlined Codex Debugging Workflow
Automated first-step debugging for Codex updates and issues.
"""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import yaml
import requests


class StreamlinedCodexDebugger:
    """Fast, automated first-step debugging for Codex issues."""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.debug_steps = []
        self.issues_found = []
        # Central flag to bypass GitHub-related checks in restricted environments
        self.skip_github_checks = bool(os.getenv("SKIP_GH_CHECK"))
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("CODEX_TOKEN")
        if token and not os.getenv("GH_TOKEN"):
            os.environ["GH_TOKEN"] = token
        
    def run_command(self, cmd: str, timeout: int = 30) -> Tuple[bool, str]:
        """Run command with timeout and capture output.

        Parses the command string with ``shlex`` for safety and only allows
        whitelisted commands.
        """
        # Whitelist of allowed commands (add more as needed)
        ALLOWED_COMMANDS = {"gh", "git"}
        try:
            # Parse command string into arguments to avoid shell injection
            args = shlex.split(cmd)
            if not args or args[0] not in ALLOWED_COMMANDS:
                return False, f"Command '{args[0] if args else cmd}' is not allowed."
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, str(e)
    
    def log_step(self, step: str, status: str, details: str = ""):
        """Log debugging step with status."""
        self.debug_steps.append({
            "step": step,
            "status": status,
            "details": details,
            "timestamp": time.strftime("%H:%M:%S")
        })
        print(f"[{time.strftime('%H:%M:%S')}] {status}: {step}")
        if details:
            print(f"    {details}")
    
    def check_github_integration(self) -> bool:
        """Quick check of GitHub integration status."""
        if self.skip_github_checks:
            self.log_step(
                "GitHub Integration Check",
                "ℹ️  SKIPPED",
                "SKIP_GH_CHECK set; skipping GitHub checks",
            )
            return True

        self.log_step("GitHub Integration Check", "RUNNING")

        # Check if we can access GitHub API
        success, output = self.run_command("gh auth status", timeout=10)
        if not success:
            self.log_step(
                "GitHub Auth",
                "⚠️  WARNING",
                "GitHub CLI not authenticated - skipping further GitHub checks",
            )
            return True

        # Check current PR status if in PR context
        success, output = self.run_command(
            "gh pr status --json number,title", timeout=15
        )
        if success and output.strip():
            try:
                pr_data = json.loads(output)
                if pr_data.get("currentBranch"):
                    pr_info = pr_data["currentBranch"]
                    self.log_step(
                        "GitHub PR Status",
                        "✅ SUCCESS",
                        f"PR #{pr_info.get('number', 'N/A')}: {pr_info.get('title', 'N/A')}",
                    )
                else:
                    self.log_step(
                        "GitHub PR Status",
                        "ℹ️  INFO",
                        "No active PR on current branch",
                    )
            except json.JSONDecodeError:
                self.log_step(
                    "GitHub PR Status",
                    "⚠️  WARNING",
                    "Could not parse PR data",
                )

        self.log_step("GitHub Integration Check", "✅ COMPLETE")
        return True
    
    def check_workflow_permissions(self) -> bool:
        """Check GitHub Actions workflow permissions."""
        self.log_step("Workflow Permissions Check", "RUNNING")
        
        workflow_file = self.repo_root / ".github/workflows/codex-auto-debug.yml"
        if not workflow_file.exists():
            self.issues_found.append("Codex auto-debug workflow missing")
            self.log_step("Workflow File", "❌ FAILED", "codex-auto-debug.yml not found")
            return False
        
        # Validate permissions using YAML parsing
        try:
            workflow_data = yaml.safe_load(workflow_file.read_text())
        except yaml.YAMLError:
            self.issues_found.append("Invalid workflow YAML format")
            self.log_step("Workflow Permissions", "❌ FAILED", "Invalid YAML")
            return False

        required_perms = {
            "contents": "write",
            "pull-requests": "write",
            "issues": "write",
        }

        permissions = workflow_data.get("permissions")
        if isinstance(permissions, dict):
            missing_perms = [
                f"{perm}: {value}"
                for perm, value in required_perms.items()
                if permissions.get(perm) != value
            ]

            if missing_perms:
                self.issues_found.append(
                    f"Missing permissions: {', '.join(missing_perms)}"
                )
                self.log_step(
                    "Workflow Permissions",
                    "❌ FAILED",
                    f"Missing: {missing_perms}",
                )
                return False

            self.log_step(
                "Workflow Permissions",
                "✅ SUCCESS",
                "All required permissions present",
            )
            return True

        # Fall back to job-level permissions if no top-level block

        jobs = workflow_data.get("jobs")
        if not isinstance(jobs, dict):
            self.issues_found.append("Workflow missing permissions block")
            self.log_step(
                "Workflow Permissions",
                "❌ FAILED",
                "Missing permissions block",
            )
            return False

        job_issues = []
        for job_name, job_data in jobs.items():
            job_perms = job_data.get("permissions")
            if not isinstance(job_perms, dict):
                job_issues.append(f"{job_name}: missing permissions block")
                continue
            missing_perms = [
                f"{perm}: {value}"
                for perm, value in required_perms.items()
                if job_perms.get(perm) != value
            ]
            if missing_perms:
                job_issues.append(f"{job_name}: missing {', '.join(missing_perms)}")

        if job_issues:
            issue_text = "; ".join(job_issues)
            self.issues_found.append(f"Job-level permissions issues: {issue_text}")
            self.log_step(
                "Workflow Permissions",
                "❌ FAILED",
                f"Jobs with issues: {issue_text}",
            )
            return False

        self.log_step(
            "Workflow Permissions",
            "✅ SUCCESS",
            "All required job-level permissions present",
        )
        return True
    
    def check_recent_workflow_runs(self) -> bool:
        """Check recent workflow run status."""
        if self.skip_github_checks:
            self.log_step(
                "Recent Workflow Runs",
                "ℹ️  SKIPPED",
                "SKIP_GH_CHECK set; skipping GitHub checks",
            )
            return True

        self.log_step("Recent Workflow Runs", "RUNNING")

        success, output = self.run_command(
            "gh run list --limit 3 --json databaseId,conclusion,workflowName",
            timeout=20,
        )
        if not success:
            self.log_step(
                "Workflow Runs", "⚠️  WARNING", "Could not fetch workflow runs"
            )
            return True
        
        try:
            runs = json.loads(output)
            codex_runs = [run for run in runs if "Codex" in run.get("workflowName", "")]
            
            if not codex_runs:
                self.log_step("Codex Workflows", "ℹ️  INFO", "No recent Codex workflow runs")
                return True
            
            latest_run = codex_runs[0]
            conclusion = latest_run.get("conclusion", "unknown")
            
            if conclusion == "failure":
                self.issues_found.append("Latest Codex workflow failed")
                self.log_step("Latest Codex Run", "❌ FAILED", f"Run ID: {latest_run.get('databaseId')}")
                
                # Get failure details
                run_id = latest_run.get("databaseId")
                success, log_output = self.run_command(f"gh run view {run_id} --json jobs", timeout=15)
                if success:
                    jobs_data = json.loads(log_output)
                    failed_jobs = [job for job in jobs_data.get("jobs", []) if job.get("conclusion") == "failure"]
                    if failed_jobs:
                        failed_steps = []
                        for job in failed_jobs:
                            for step in job.get("steps", []):
                                if step.get("conclusion") == "failure":
                                    failed_steps.append(step.get("name", "unknown"))
                        
                        if failed_steps:
                            self.log_step("Failed Steps", "❌ DETAILS", f"Steps: {', '.join(failed_steps)}")
                
                return False
            else:
                self.log_step("Latest Codex Run", "✅ SUCCESS", f"Status: {conclusion}")
        
        except json.JSONDecodeError:
            self.log_step("Workflow Runs", "⚠️  WARNING", "Could not parse workflow data")
        
        return True
    
    def check_branch_status(self) -> bool:
        """Check current branch and its status."""
        self.log_step("Branch Status Check", "RUNNING")
        
        # Get current branch
        success, branch = self.run_command("git branch --show-current")
        if not success:
            self.issues_found.append("Could not determine current branch")
            self.log_step("Current Branch", "❌ FAILED", "Git branch check failed")
            return False
        
        branch = branch.strip()
        self.log_step("Current Branch", "ℹ️  INFO", f"Branch: {branch}")
        
        # Check if it's a Codex branch
        if not branch.startswith("codex/"):
            self.log_step("Branch Type", "ℹ️  INFO", "Not a Codex branch - workflow won't trigger")
        else:
            self.log_step("Branch Type", "✅ SUCCESS", "Codex branch - workflow will trigger")
        
        # Check if branch is up to date with remote
        success, output = self.run_command("git status --porcelain -b")
        if success and "ahead" in output:
            self.log_step("Branch Sync", "ℹ️  INFO", "Branch has unpushed commits")
        elif success and "behind" in output:
            self.log_step("Branch Sync", "⚠️  WARNING", "Branch is behind remote")
        else:
            self.log_step("Branch Sync", "✅ SUCCESS", "Branch is in sync")
        
        return True
    
    def quick_test_permissions(self) -> bool:
        """Quick test of GitHub permissions without creating a real PR."""
        if self.skip_github_checks:
            self.log_step(
                "Quick Permissions Test",
                "ℹ️  SKIPPED",
                "SKIP_GH_CHECK set; skipping GitHub checks",
            )
            return True

        self.log_step("Quick Permissions Test", "RUNNING")

        # Test if we can access repository info
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("CODEX_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        repo = os.getenv("GITHUB_REPOSITORY")
        if not repo:
            self.issues_found.append("GITHUB_REPOSITORY is not set or is empty")
            self.log_step(
                "Repository Access",
                "❌ FAILED",
                "GITHUB_REPOSITORY environment variable is missing or empty",
            )
            return False

        # Validate repo format: must be 'owner/repo'
        parts = repo.split("/")
        if len(parts) != 2 or not all(parts) or any("/" in part for part in parts):
            self.issues_found.append("Invalid GITHUB_REPOSITORY format")
            self.log_step(
                "Repository Access",
                "❌ FAILED",
                "GITHUB_REPOSITORY must be in 'owner/repo' format",
            )
            return False

        api_url = f"https://api.github.com/repos/{repo}"
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code != 200:
                self.issues_found.append("Cannot access repository info")
                error_details = f"HTTP {response.status_code}: {response.text.strip() or response.reason}"
                self.log_step(
                    "Repository Access",
                    "❌ FAILED",
                    f"Check repository permissions. {error_details}",
                )
                return False
            repo_data = response.json()
            repo_name = repo_data.get("full_name", repo or "unknown")
            self.log_step(
                "Repository Access",
                "✅ SUCCESS",
                f"Repository: {repo_name}",
            )
        except Exception as e:
            self.issues_found.append("Cannot access repository info")
            self.log_step("Repository Access", "❌ FAILED", str(e))
            return False
        
        # Test if we can list workflow runs (requires actions:read)
        success, output = self.run_command("gh run list --limit 1", timeout=10)
        if success:
            self.log_step("Actions Access", "✅ SUCCESS", "Can access workflow runs")
        else:
            self.issues_found.append("Cannot access GitHub Actions")
            self.log_step("Actions Access", "❌ FAILED", "Missing actions:read permission")
            return False
        
        return True
    
    def generate_report(self) -> str:
        """Generate comprehensive debugging report."""
        report = ["# 🔍 Streamlined Codex Debugging Report", ""]
        report.append(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Repository**: {self.repo_root.name}")
        report.append("")
        
        # Summary
        if self.issues_found:
            report.append("## ❌ Issues Found")
            for issue in self.issues_found:
                report.append(f"- {issue}")
            report.append("")
        else:
            report.append("## ✅ All Checks Passed")
            report.append("No issues detected in streamlined debugging.")
            report.append("")
        
        # Detailed steps
        report.append("## 📋 Debugging Steps")
        for step in self.debug_steps:
            status = step["status"]
            if "SUCCESS" in status:
                status_icon = "✅"
            elif "FAILED" in status:
                status_icon = "❌"
            elif "SKIPPED" in status or "INFO" in status:
                status_icon = "ℹ️"
            else:
                status_icon = "⚠️"
            report.append(f"**{step['timestamp']}** {status_icon} {step['step']}")
            if step["details"]:
                report.append(f"  - {step['details']}")
            report.append("")
        
        # Quick fixes
        if self.issues_found:
            report.append("## 🛠️ Quick Fixes")
            for issue in self.issues_found:
                if "GitHub CLI not authenticated" in issue:
                    report.append("- Run: `gh auth login`")
                elif "Workflow missing permissions" in issue:
                    report.append("- Add permissions block to `.github/workflows/codex-auto-debug.yml`")
                elif "Latest Codex workflow failed" in issue:
                    report.append("- Check workflow logs: `gh run view --log-failed`")
                elif "Cannot access repository info" in issue:
                    report.append("- Check GitHub token permissions")
            report.append("")
        
        return "\n".join(report)
    
    def run_streamlined_debug(self) -> bool:
        """Run complete streamlined debugging workflow."""
        print("🚀 Starting Streamlined Codex Debugging...")
        print()
        
        checks = [
            self.check_github_integration,
            self.check_branch_status,
            self.check_workflow_permissions,
            self.quick_test_permissions,
            self.check_recent_workflow_runs,
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                self.issues_found.append(f"Check failed with error: {str(e)}")
                all_passed = False
            print()  # Space between checks
        
        # Generate and save report
        report = self.generate_report()
        
        print("📊 DEBUGGING COMPLETE")
        print("=" * 50)
        print(report)
        
        # Save report to file
        report_file = self.repo_root / "streamlined_debug_report.md"
        report_file.write_text(report)
        print(f"\n💾 Report saved to: {report_file}")
        
        return all_passed


def main():
    """Main entry point for streamlined debugging."""
    debugger = StreamlinedCodexDebugger()
    success = debugger.run_streamlined_debug()
    
    if success:
        print("\n🎉 All checks passed! Codex integration should work correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  Found {len(debugger.issues_found)} issues that need attention.")
        sys.exit(1)


if __name__ == "__main__":
    main()
