#!/usr/bin/env python3
"""
Automated debugging workflow for Codex Pull Requests.
Runs comprehensive checks and fixes common integration issues.
"""

import subprocess
import sys
import json
import os
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
            
            # Look for the pattern where fin_rngs dict is passed to sweep function
            if "run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)" in content:
                # Fix by converting dict to list
                new_content = content.replace(
                    "run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)",
                    "fin_rngs_list = list(fin_rngs.values())\n        results = run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs_list)"
                ).replace(
                    "results = run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs_list)",
                    "run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs_list)"
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
        # Fix DataFrame/Series type issues
        cli_file = self.repo_root / "pa_core" / "cli.py"
        if cli_file.exists():
            content = cli_file.read_text()
            
            # Pattern 1: Ensure idx_series is Series
            if "idx_series = load_index_returns" in content:
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
        print("� Starting iterative Codex PR debugging...")
        
        iteration = 0
        all_fixes_applied = []
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            
            # Comprehensive issue detection
            all_issues = []
            all_issues.extend(self.check_dependencies())
            all_issues.extend(self.check_imports_and_types())
            all_issues.extend(self.check_code_style())
            all_issues.extend(self.check_tests())
            all_issues.extend(self.check_ci_cd_compliance())  # NEW: CI/CD specific checks
            
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
            report.append("")
            
            report.append("## 📋 Manual Intervention Required")
            report.append("The following issues need manual review:")
            for issue in results['final_issues']:
                if issue['severity'] == 'high':
                    report.append(f"1. **{issue['type']}**: {issue['description']}")
            report.append("")
        
        # Status summary
        if results['ci_cd_ready']:
            report.append("## 🎉 CI/CD Status: READY")
            report.append("All automated checks pass. Branch is ready for CI/CD pipeline.")
        else:
            report.append("## ❌ CI/CD Status: ISSUES REMAIN")
            report.append(f"Found {len(results['final_issues'])} issues that prevent CI/CD success.")
            report.append("Manual fixes required before CI/CD will pass.")
        
        report.append("")
        report.append("## 📋 Next Steps")
        if results['ci_cd_ready']:
            report.append("1. Commit any remaining changes")
            report.append("2. Push to trigger CI/CD pipeline")
            report.append("3. Monitor pipeline for successful completion")
        else:
            report.append("1. Review and fix remaining high-severity issues")
            report.append("2. Re-run debugging: `make debug-codex`")
            report.append("3. Repeat until CI/CD ready")
        
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
