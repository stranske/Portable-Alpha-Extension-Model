#!/usr/bin/env python3
"""
Streamlined Codex Debugging Workflow
Automated first-step debugging for Codex updates and issues.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple


class StreamlinedCodexDebugger:
    """Fast, automated first-step debugging for Codex issues."""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.debug_steps = []
        self.issues_found = []
        
    def run_command(self, cmd: str, timeout: int = 30) -> Tuple[bool, str]:
        """Run command with timeout and capture output."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
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
        self.log_step("GitHub Integration Check", "RUNNING")
        
        # Check if we can access GitHub API
        success, output = self.run_command("gh auth status", timeout=10)
        if not success:
            self.issues_found.append("GitHub CLI not authenticated")
            self.log_step("GitHub Auth", "❌ FAILED", "Run: gh auth login")
            return False
            
        # Check current PR status if in PR context
        success, output = self.run_command("gh pr status --json number,title", timeout=15)
        if success and output.strip():
            try:
                pr_data = json.loads(output)
                if pr_data.get("currentBranch"):
                    pr_info = pr_data["currentBranch"]
                    self.log_step("GitHub PR Status", "✅ SUCCESS", 
                                f"PR #{pr_info.get('number', 'N/A')}: {pr_info.get('title', 'N/A')}")
                else:
                    self.log_step("GitHub PR Status", "⚠️  INFO", "No active PR on current branch")
            except json.JSONDecodeError:
                self.log_step("GitHub PR Status", "⚠️  WARNING", "Could not parse PR data")
        
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
        
        # Check for permissions block
        content = workflow_file.read_text()
        if "permissions:" not in content:
            self.issues_found.append("Workflow missing permissions block")
            self.log_step("Workflow Permissions", "❌ FAILED", "Missing permissions block")
            return False
        
        required_perms = ["contents: write", "pull-requests: write", "issues: write"]
        missing_perms = [perm for perm in required_perms if perm not in content]
        
        if missing_perms:
            self.issues_found.append(f"Missing permissions: {', '.join(missing_perms)}")
            self.log_step("Workflow Permissions", "❌ FAILED", f"Missing: {missing_perms}")
            return False
        
        self.log_step("Workflow Permissions", "✅ SUCCESS", "All required permissions present")
        return True
    
    def check_recent_workflow_runs(self) -> bool:
        """Check recent workflow run status."""
        self.log_step("Recent Workflow Runs", "RUNNING")
        
        success, output = self.run_command("gh run list --limit 3 --json databaseId,conclusion,workflowName", timeout=20)
        if not success:
            self.log_step("Workflow Runs", "⚠️  WARNING", "Could not fetch workflow runs")
            return False
        
        try:
            runs = json.loads(output)
            codex_runs = [run for run in runs if "Codex" in run.get("workflowName", "")]
            
            if not codex_runs:
                self.log_step("Codex Workflows", "⚠️  INFO", "No recent Codex workflow runs")
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
        self.log_step("Current Branch", "✅ INFO", f"Branch: {branch}")
        
        # Check if it's a Codex branch
        if not branch.startswith("codex/"):
            self.log_step("Branch Type", "⚠️  INFO", "Not a Codex branch - workflow won't trigger")
        else:
            self.log_step("Branch Type", "✅ SUCCESS", "Codex branch - workflow will trigger")
        
        # Check if branch is up to date with remote
        success, output = self.run_command("git status --porcelain -b")
        if success and "ahead" in output:
            self.log_step("Branch Sync", "⚠️  INFO", "Branch has unpushed commits")
        elif success and "behind" in output:
            self.log_step("Branch Sync", "⚠️  WARNING", "Branch is behind remote")
        else:
            self.log_step("Branch Sync", "✅ SUCCESS", "Branch is in sync")
        
        return True
    
    def quick_test_permissions(self) -> bool:
        """Quick test of GitHub permissions without creating a real PR."""
        self.log_step("Quick Permissions Test", "RUNNING")
        
        # Test if we can access repository info
        success, output = self.run_command("gh repo view --json name,owner", timeout=10)
        if not success:
            self.issues_found.append("Cannot access repository info")
            self.log_step("Repository Access", "❌ FAILED", "Check repository permissions")
            return False
        
        try:
            repo_data = json.loads(output)
            repo_name = f"{repo_data['owner']['login']}/{repo_data['name']}"
            self.log_step("Repository Access", "✅ SUCCESS", f"Repository: {repo_name}")
        except (json.JSONDecodeError, KeyError):
            self.log_step("Repository Access", "⚠️  WARNING", "Could not parse repository data")
        
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
            status_icon = "✅" if "SUCCESS" in step["status"] else "❌" if "FAILED" in step["status"] else "⚠️"
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
