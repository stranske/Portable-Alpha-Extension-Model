#!/usr/bin/env python3
"""
Automated debugging workflow for Codex Pull Requests.
Runs comprehensive checks and fixes common integration issues.
"""

import subprocess
import sys
import json
import os
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class CodexPRDebugger:
    """Automated debugging workflow for Codex implementations."""
    
    def __init__(self, branch_name: str = None):
        self.branch_name = branch_name
        self.repo_root = Path.cwd()
        self.issues_found: List[Dict] = []
        self.fixes_applied: List[str] = []
        
    def run_command(self, cmd: str, capture_output: bool = True) -> Tuple[bool, str]:
        """Run shell command and return success status and output."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=capture_output, text=True
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def check_github_ci_cd_status(self) -> List[Dict]:
        """Check actual GitHub CI/CD pipeline status and errors."""
        issues = []
        
        try:
            # Get GitHub token from environment (if available)
            github_token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
            
            # Try to get PR status using GitHub CLI first
            success, gh_output = self.run_command(
                f"gh pr status --json statusCheckRollup 2>/dev/null || echo 'gh-cli-unavailable'"
            )
            
            if "gh-cli-unavailable" not in gh_output:
                try:
                    status_data = json.loads(gh_output)
                    if 'currentBranch' in status_data and 'statusCheckRollup' in status_data['currentBranch']:
                        for check in status_data['currentBranch']['statusCheckRollup']:
                            if check['state'] in ['FAILURE', 'ERROR']:
                                issues.append({
                                    'type': 'ci_cd_failure',
                                    'description': f"CI/CD Check Failed: {check['context']} - {check.get('description', 'No description')}",
                                    'severity': 'critical',
                                    'github_url': check.get('targetUrl', ''),
                                    'context': check['context']
                                })
                            elif check['state'] == 'PENDING':
                                issues.append({
                                    'type': 'ci_cd_pending',
                                    'description': f"CI/CD Check Running: {check['context']}",
                                    'severity': 'info',
                                    'github_url': check.get('targetUrl', ''),
                                    'context': check['context']
                                })
                except json.JSONDecodeError:
                    pass
            
            # Alternative: Try to get repo info and infer CI/CD status
            success, git_output = self.run_command("git remote get-url origin")
            if success and "github.com" in git_output:
                # Extract repo info
                repo_url = git_output.strip()
                if repo_url.endswith('.git'):
                    repo_url = repo_url[:-4]
                
                # Try to get recent workflow runs (requires GitHub API)
                issues.append({
                    'type': 'ci_cd_info',
                    'description': f"GitHub repository detected: {repo_url}",
                    'severity': 'info',
                    'github_url': f"{repo_url}/actions"
                })
        
        except Exception as e:
            issues.append({
                'type': 'ci_cd_check_error',
                'description': f"Failed to check CI/CD status: {str(e)}",
                'severity': 'warning'
            })
        
        return issues
    
    def analyze_github_actions_logs(self, check_context: str) -> Dict:
        """Analyze specific GitHub Actions log for common error patterns."""
        patterns = {
            'Type Checking': [
                r'error: .*Cannot assign.*',
                r'error: .*Incompatible.*',
                r'error: .*has no attribute.*',
            ],
            'Tests': [
                r'FAILED.*::(.*)',
                r'ImportError:.*',
                r'TypeError:.*',
            ],
            'Code Quality': [
                r'E\d+.*line too long.*',
                r'F\d+.*imported but unused.*',
                r'W\d+.*',
            ]
        }
        
        analysis = {
            'context': check_context,
            'error_patterns': patterns.get(check_context, []),
            'suggested_fixes': []
        }
        
        # Map common CI/CD failures to local debugging commands
        if check_context == 'Type Checking':
            analysis['local_debug_cmd'] = "dev-env/bin/python -m mypy pa_core/ --strict"
            analysis['suggested_fixes'] = [
                "Check type annotations in recently modified files",
                "Ensure imports match actual function signatures",
                "Verify DataFrame/Series type conversions"
            ]
        elif 'Tests' in check_context:
            analysis['local_debug_cmd'] = "dev-env/bin/python -m pytest tests/ -v --tb=short"
            analysis['suggested_fixes'] = [
                "Run tests locally to reproduce failures",
                "Check for missing imports or dependencies",
                "Verify test data and fixtures"
            ]
        elif check_context == 'Code Quality':
            analysis['local_debug_cmd'] = "dev-env/bin/ruff format --check pa_core/ tests/ dashboard/"
            analysis['suggested_fixes'] = [
                "Run code formatting: dev-env/bin/ruff format pa_core/ tests/ dashboard/",
                "Check line length and import ordering",
                "Remove unused imports"
            ]
        
        return analysis
    
    def check_imports_and_types(self) -> List[Dict]:
        """Check for import and type issues."""
        issues = []
        
        # Run mypy type checking
        success, output = self.run_command("make typecheck")
        if not success and "error:" in output.lower():
            for line in output.split('\n'):
                if 'error:' in line:
                    issues.append({
                        'type': 'type_error',
                        'description': line.strip(),
                        'severity': 'high'
                    })
        
        # Check for unused imports with flake8
        success, output = self.run_command("dev-env/bin/flake8 pa_core/ --select=F401,F811")
        if not success:
            for line in output.split('\n'):
                if 'F401' in line or 'F811' in line:
                    issues.append({
                        'type': 'unused_import',
                        'description': line.strip(),
                        'severity': 'medium'
                    })
        
        return issues
    
    def check_code_style(self) -> List[Dict]:
        """Check for code style violations."""
        issues = []
        
        # Check formatting with black
        success, output = self.run_command("dev-env/bin/black --check pa_core/ tests/ dashboard/")
        if not success:
            issues.append({
                'type': 'formatting',
                'description': 'Code formatting issues detected',
                'severity': 'medium'
            })
        
        # Check import ordering with isort
        success, output = self.run_command("dev-env/bin/isort --check-only pa_core/ tests/ dashboard/")
        if not success:
            issues.append({
                'type': 'import_order',
                'description': 'Import ordering issues detected',
                'severity': 'medium'
            })
        
        # Check line length and other style issues
        success, output = self.run_command("dev-env/bin/flake8 pa_core/ tests/ dashboard/ --max-line-length=88 --ignore=E203,W503")
        if not success:
            issues.append({
                'type': 'style_violations',
                'description': f'Style violations found:\n{output}',
                'severity': 'medium'
            })
        
        return issues
    
    def check_tests(self) -> List[Dict]:
        """Check if tests pass."""
        issues = []
        
        success, output = self.run_command("python -m pytest tests/ -x --tb=short")
        if not success:
            issues.append({
                'type': 'test_failure',
                'description': f'Tests failing:\n{output}',
                'severity': 'high'
            })
        
        return issues
    
    def check_dependencies(self) -> List[Dict]:
        """Check for missing dependencies or version conflicts."""
        issues = []
        
        # Try importing key modules
        import_tests = [
            "from pa_core.sweep import run_parameter_sweep",
            "from pa_core.reporting.sweep_excel import export_sweep_results",
            "import pa_core.config"
        ]
        
        for import_test in import_tests:
            success, output = self.run_command(f"python -c \"{import_test}\"")
            if not success:
                issues.append({
                    'type': 'import_error',
                    'description': f'Import failed: {import_test}\n{output}',
                    'severity': 'high'
                })
        
        return issues
    
    def auto_fix_formatting(self) -> bool:
        """Automatically fix code formatting issues."""
        print("🎨 Auto-fixing code formatting...")
        
        # Run black formatter
        success1, _ = self.run_command("dev-env/bin/black pa_core/ tests/ dashboard/")
        
        # Fix import ordering
        success2, _ = self.run_command("dev-env/bin/isort pa_core/ tests/ dashboard/")
        
        if success1 and success2:
            self.fixes_applied.append("Applied code formatting fixes")
            return True
        return False
    
    def auto_fix_common_issues(self) -> None:
        """Apply common fixes for typical Codex integration issues."""
        print("🔧 Checking for common integration issues...")
        
        # Check for dict vs list type mismatches (like the fin_rngs issue)
        self._fix_fin_rngs_type_issue()
        
        # Remove common unused imports
        self._remove_unused_imports()
        
        # Fix empty init files
        self._fix_empty_init_files()
    
    def _fix_fin_rngs_type_issue(self) -> None:
        """Fix the specific fin_rngs dict->list conversion issue."""
        cli_file = self.repo_root / "pa_core" / "cli.py"
        if cli_file.exists():
            content = cli_file.read_text()
            
            # Look for the pattern where fin_rngs is incorrectly converted to list
            # The correct fix is to keep fin_rngs as dict and capture return value
            if "fin_rngs_list = list(fin_rngs.values())" in content:
                # Fix by removing the list conversion and properly capturing results
                new_content = content.replace(
                    "results = fin_rngs_list = list(fin_rngs.values())\n        run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs_list)",
                    "results = run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)"
                ).replace(
                    "fin_rngs_list = list(fin_rngs.values())\n        results = run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs_list)",
                    "results = run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)"
                )
                
                if new_content != content:
                    cli_file.write_text(new_content)
                    self.fixes_applied.append("Fixed fin_rngs type conversion issue")
    
    def _remove_unused_imports(self) -> None:
        """Remove commonly unused imports that Codex adds."""
        files_to_check = [
            self.repo_root / "pa_core" / "cli.py",
            self.repo_root / "pa_core" / "__init__.py"
        ]
        
        unused_patterns = [
            "import numpy.typing as npt",
            "from numpy.typing import NDArray",
            "from rich.table import Table"  # if not used
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                content = file_path.read_text()
                original_content = content
                
                for pattern in unused_patterns:
                    if pattern in content:
                        # Simple removal - could be enhanced with AST analysis
                        content = content.replace(f"{pattern}\n", "")
                        content = content.replace(pattern, "")
                
                if content != original_content:
                    file_path.write_text(content)
                    self.fixes_applied.append(f"Removed unused imports from {file_path.name}")
    
    def _fix_empty_init_files(self) -> None:
        """Fix empty or whitespace-only __init__.py files."""
        init_files = list(self.repo_root.rglob("__init__.py"))
        
        for init_file in init_files:
            content = init_file.read_text().strip()
            if not content or content.isspace():
                # Add a basic docstring
                module_name = init_file.parent.name.replace("_", " ").title()
                new_content = f'"""{module_name} module for Portable Alpha Extension Model."""\n'
                init_file.write_text(new_content)
                self.fixes_applied.append(f"Fixed empty {init_file.relative_to(self.repo_root)}")
    
    def check_ci_cd_compliance(self) -> List[Dict]:
        """Check exact CI/CD pipeline compliance."""
        issues = []
        
        # 1. Check Ruff formatting (exact CI/CD command)
        success, output = self.run_command("dev-env/bin/ruff format --check pa_core/ tests/ dashboard/")
        if not success:
            issues.append({
                'type': 'ruff_formatting',
                'description': f'Ruff formatting issues:\n{output}',
                'severity': 'high'
            })
        
        # 2. Check exact mypy as CI/CD runs it
        success, output = self.run_command("dev-env/bin/python -m mypy pa_core/ --strict")
        if not success and "error:" in output.lower():
            for line in output.split('\n'):
                if 'error:' in line and line.strip():
                    issues.append({
                        'type': 'ci_type_error',
                        'description': line.strip(),
                        'severity': 'high'
                    })
        
        # 3. Check pytest exactly like CI/CD
        success, output = self.run_command("dev-env/bin/python -m pytest tests/ -v --tb=short")
        if not success:
            issues.append({
                'type': 'ci_test_failure',
                'description': f'CI/CD test failures:\n{output}',
                'severity': 'high'
            })
        
        # 4. Validate devcontainer.json (CI/CD requirement)
        success, output = self.run_command("python -m json.tool .devcontainer/devcontainer.json")
        if not success:
            issues.append({
                'type': 'devcontainer_json',
                'description': f'Invalid devcontainer.json:\n{output}',
                'severity': 'medium'
            })
        
        return issues
    
    def auto_fix_ci_cd_issues(self) -> None:
        """Apply fixes specifically for CI/CD compliance."""
        # Fix Ruff formatting issues
        success, output = self.run_command("dev-env/bin/ruff format pa_core/ tests/ dashboard/")
        if success:
            self.fixes_applied.append("Applied Ruff formatting fixes")
        
        # Fix type errors with common patterns
        self._fix_type_errors()
        
        # Fix devcontainer.json if needed
        self._fix_devcontainer_json()
    
    def _fix_type_errors(self) -> None:
        """Fix common type errors that break CI/CD."""
        # Fix DataFrame/Series type issues (but only if not already present)
        cli_file = self.repo_root / "pa_core" / "cli.py"
        if cli_file.exists():
            content = cli_file.read_text()
            
            # Only add type checking if it's not already present
            if "idx_series = load_index_returns" in content and "# Ensure idx_series is a pandas Series for type safety" not in content:
                type_fix = '''idx_series = load_index_returns(args.index)
    
    # Ensure idx_series is a pandas Series for type safety
    if isinstance(idx_series, pd.DataFrame):
        idx_series = idx_series.squeeze()
        if not isinstance(idx_series, pd.Series):
            raise ValueError("Index data must be convertible to pandas Series")
    elif not isinstance(idx_series, pd.Series):
        raise ValueError("Index data must be a pandas Series")'''
                
                content = content.replace(
                    "idx_series = load_index_returns(args.index)",
                    type_fix
                )
                cli_file.write_text(content)
                self.fixes_applied.append("Fixed DataFrame/Series type issue in cli.py")
        
        # Fix mapping vs list issues in sweep.py
        sweep_file = self.repo_root / "pa_core" / "sweep.py"
        if sweep_file.exists():
            content = sweep_file.read_text()
            
            # Fix function signature to match actual usage
            old_sig = "fin_rngs: List[np.random.Generator]"
            new_sig = "fin_rngs: Dict[str, np.random.Generator]"
            if old_sig in content:
                content = content.replace(old_sig, new_sig)
                sweep_file.write_text(content)
                self.fixes_applied.append("Fixed fin_rngs type signature in sweep.py")
    
    def _fix_devcontainer_json(self) -> None:
        """Fix devcontainer.json formatting issues."""
        devcontainer_file = self.repo_root / ".devcontainer" / "devcontainer.json"
        if devcontainer_file.exists():
            try:
                import json
                content = devcontainer_file.read_text()
                # Try to parse and reformat with proper JSON
                data = json.loads(content)
                formatted = json.dumps(data, indent=2, ensure_ascii=False)
                devcontainer_file.write_text(formatted)
                self.fixes_applied.append("Fixed devcontainer.json formatting")
            except json.JSONDecodeError:
                # Fix common JSON issues
                content = devcontainer_file.read_text()
                # Replace single quotes with double quotes
                content = content.replace("'", '"')
                try:
                    data = json.loads(content)
                    formatted = json.dumps(data, indent=2, ensure_ascii=False)
                    devcontainer_file.write_text(formatted)
                    self.fixes_applied.append("Fixed devcontainer.json quote issues")
                except:
                    pass
    
    def run_iterative_check(self, max_iterations: int = 3) -> Dict:
        """Run debugging with iteration until all CI/CD issues are resolved."""
        print("🔍 Starting iterative Codex PR debugging...")
        
        # First, check actual GitHub CI/CD status
        print("🔗 Checking GitHub CI/CD pipeline status...")
        github_issues = self.check_github_ci_cd_status()
        if github_issues:
            print(f"📊 Found {len(github_issues)} GitHub CI/CD status items:")
            for issue in github_issues:
                emoji = "❌" if issue['severity'] in ['critical', 'high'] else "⚠️" if issue['severity'] == 'warning' else "ℹ️"
                print(f"  {emoji} {issue['description']}")
                if 'github_url' in issue and issue['github_url']:
                    print(f"    🔗 {issue['github_url']}")
        
        iteration = 0
        all_fixes_applied = []
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            
            # Comprehensive issue detection (including GitHub CI/CD status)
            all_issues = []
            all_issues.extend(github_issues)  # Include GitHub CI/CD status
            all_issues.extend(self.check_dependencies())
            all_issues.extend(self.check_imports_and_types())
            all_issues.extend(self.check_code_style())
            all_issues.extend(self.check_tests())
            all_issues.extend(self.check_ci_cd_compliance())  # Local CI/CD checks
            
            if not all_issues:
                print("✅ No issues found!")
                break
            
            print(f"🔧 Found {len(all_issues)} issues, applying fixes...")
            
            # Reset fixes for this iteration
            self.fixes_applied = []
            
            # Apply comprehensive fixes
            self.auto_fix_common_issues()
            self.auto_fix_formatting()
            self.auto_fix_ci_cd_issues()  # NEW: CI/CD specific fixes
            
            # Collect fixes
            if self.fixes_applied:
                all_fixes_applied.extend(self.fixes_applied)
                print(f"🔧 Applied {len(self.fixes_applied)} fixes")
            else:
                print("⚠️ No automatic fixes available for remaining issues")
                break
        
        # Final validation
        print("\n🔄 Final CI/CD compliance check...")
        final_issues = self.check_ci_cd_compliance()
        
        return {
            'iterations': iteration,
            'total_fixes_applied': all_fixes_applied,
            'final_issues': final_issues,
            'success': len(final_issues) == 0,
            'ci_cd_ready': len(final_issues) == 0
        }
    
    def generate_enhanced_report(self, results: Dict) -> str:
        """Generate enhanced report for iterative debugging results."""
        report = []
        
        report.append("# 🔍 Enhanced Codex PR Debugging Report")
        report.append(f"Branch: {self.branch_name}")
        report.append(f"Iterations: {results['iterations']}")
        report.append("")
        
        # Add GitHub CI/CD status section
        github_issues = [issue for issue in results.get('all_issues', []) if issue.get('type', '').startswith('ci_cd_')]
        if github_issues:
            report.append("## 🔗 GitHub CI/CD Pipeline Status")
            for issue in github_issues:
                if issue['severity'] == 'critical':
                    emoji = "❌"
                elif issue['severity'] == 'warning':
                    emoji = "⚠️"
                elif issue['severity'] == 'info':
                    emoji = "ℹ️"
                else:
                    emoji = "📋"
                
                report.append(f"- {emoji} **{issue.get('context', issue['type'])}**: {issue['description']}")
                if 'github_url' in issue and issue['github_url']:
                    report.append(f"  🔗 [View Details]({issue['github_url']})")
            report.append("")
        
        if results['total_fixes_applied']:
            report.append("## ✅ Fixes Applied")
            for fix in results['total_fixes_applied']:
                report.append(f"- {fix}")
            report.append("")
        
        if results['final_issues']:
            report.append("## ❌ Remaining CI/CD Issues")
            for issue in results['final_issues']:
                severity_emoji = "🔥" if issue['severity'] == 'high' else "⚠️"
                report.append(f"- **{issue['type']}** ({severity_emoji}{issue['severity']}): {issue['description']}")
                
                # Add specific debugging guidance for CI/CD failures
                if issue['type'] in ['ci_cd_failure', 'ci_type_error', 'ci_test_failure']:
                    analysis = self.analyze_github_actions_logs(issue.get('context', issue['type']))
                    if 'local_debug_cmd' in analysis:
                        report.append(f"  🔧 Local Debug: `{analysis['local_debug_cmd']}`")
                    for fix in analysis.get('suggested_fixes', []):
                        report.append(f"  💡 {fix}")
            report.append("")
            
            report.append("## 📋 Manual Intervention Required")
            report.append("The following issues need manual review:")
            for issue in results['final_issues']:
                if issue['severity'] in ['high', 'critical']:
                    report.append(f"1. **{issue['type']}**: {issue['description']}")
            report.append("")
        
        # Enhanced status summary with GitHub integration
        if results['ci_cd_ready']:
            report.append("## 🎉 CI/CD Status: READY")
            report.append("✅ All automated checks pass. Branch is ready for CI/CD pipeline.")
            report.append("🚀 GitHub Actions should succeed on next push.")
        else:
            report.append("## ❌ CI/CD Status: ISSUES REMAIN")
            report.append(f"Found {len(results['final_issues'])} issues that prevent CI/CD success.")
            report.append("❌ GitHub Actions will likely fail until these are resolved.")
            report.append("Manual fixes required before CI/CD will pass.")
        
        report.append("")
        report.append("## 📋 Next Steps")
        if results['ci_cd_ready']:
            report.append("1. Commit any remaining changes")
            report.append("2. Push to trigger CI/CD pipeline")
            report.append("3. Monitor GitHub Actions for successful completion")
            report.append("4. Check PR status checks at: https://github.com/[owner]/[repo]/pull/[number]")
        else:
            report.append("1. Review and fix remaining high-severity issues")
            report.append("2. Re-run debugging: `make debug-codex`")
            report.append("3. Monitor GitHub Actions for real-time feedback")
            report.append("4. Repeat until CI/CD ready")
        
        return "\n".join(report)

    # Keep the original method for backward compatibility
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive debugging report (legacy method)."""
        return self.generate_enhanced_report(results)
def main():
    """Main entry point for the debugging workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Codex Pull Request")
    parser.add_argument("--branch", help="Branch name being debugged")
    parser.add_argument("--report", help="Output file for debugging report")
    parser.add_argument("--commit", action="store_true", help="Auto-commit fixes")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum debugging iterations")
    
    args = parser.parse_args()
    
    debugger = CodexPRDebugger(args.branch)
    
    # Use iterative debugging for comprehensive CI/CD compliance
    results = debugger.run_iterative_check(max_iterations=args.max_iterations)
    
    # Generate and display report
    report = debugger.generate_enhanced_report(results)
    print("\n" + report)
    
    # Save report if requested
    if args.report:
        Path(args.report).write_text(report)
        print(f"\n📄 Report saved to {args.report}")
    
    # Auto-commit if requested and fixes were applied
    if args.commit and results['total_fixes_applied']:
        commit_msg = f"Auto-fix: CI/CD compliance fixes ({len(results['total_fixes_applied'])} changes)"
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", commit_msg])
        print(f"\n✅ Committed fixes: {commit_msg}")
    
    # Exit with proper CI/CD status
    if results['ci_cd_ready']:
        print("\n🎉 Branch is CI/CD ready!")
        sys.exit(0)
    else:
        print(f"\n❌ {len(results['final_issues'])} CI/CD issues remain")
        sys.exit(1)


if __name__ == "__main__":
    main()
