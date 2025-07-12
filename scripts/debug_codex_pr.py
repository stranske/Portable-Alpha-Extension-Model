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
        success, output = self.run_command("dev-env/bin/flake8 pa_core/ tests/ dashboard/ --max-line-length=88")
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
    
    def run_full_check(self) -> Dict:
        """Run comprehensive debugging check."""
        print("🚀 Starting automated Codex PR debugging...")
        
        # Collect all issues
        all_issues = []
        all_issues.extend(self.check_dependencies())
        all_issues.extend(self.check_imports_and_types())
        all_issues.extend(self.check_code_style())
        all_issues.extend(self.check_tests())
        
        # Apply automatic fixes
        self.auto_fix_common_issues()
        self.auto_fix_formatting()
        
        # Re-check after fixes
        print("🔄 Re-checking after automatic fixes...")
        remaining_issues = []
        remaining_issues.extend(self.check_imports_and_types())
        remaining_issues.extend(self.check_tests())
        
        return {
            'initial_issues': all_issues,
            'fixes_applied': self.fixes_applied,
            'remaining_issues': remaining_issues,
            'success': len(remaining_issues) == 0
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive debugging report."""
        report = []
        report.append("# 🔍 Codex PR Debugging Report")
        report.append(f"Branch: {self.branch_name or 'current'}")
        report.append("")
        
        if results['initial_issues']:
            report.append("## ❌ Initial Issues Found")
            for issue in results['initial_issues']:
                report.append(f"- **{issue['type']}** ({issue['severity']}): {issue['description']}")
            report.append("")
        
        if results['fixes_applied']:
            report.append("## ✅ Automatic Fixes Applied")
            for fix in results['fixes_applied']:
                report.append(f"- {fix}")
            report.append("")
        
        if results['remaining_issues']:
            report.append("## ⚠️ Remaining Issues (Manual Intervention Required)")
            for issue in results['remaining_issues']:
                report.append(f"- **{issue['type']}** ({issue['severity']}): {issue['description']}")
            report.append("")
        else:
            report.append("## 🎉 All Issues Resolved!")
            report.append("The branch is ready for CI/CD validation.")
            report.append("")
        
        report.append("## 📋 Next Steps")
        if results['remaining_issues']:
            report.append("1. Review remaining issues above")
            report.append("2. Apply manual fixes as needed")
            report.append("3. Re-run this debugger")
            report.append("4. Commit and push when clean")
        else:
            report.append("1. Commit automatic fixes")
            report.append("2. Push to trigger CI/CD pipeline")
            report.append("3. Monitor pipeline results")
        
        return "\n".join(report)


def main():
    """Main entry point for the debugging workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Codex Pull Request")
    parser.add_argument("--branch", help="Branch name being debugged")
    parser.add_argument("--report", help="Output file for debugging report")
    parser.add_argument("--commit", action="store_true", help="Auto-commit fixes")
    
    args = parser.parse_args()
    
    debugger = CodexPRDebugger(args.branch)
    results = debugger.run_full_check()
    
    # Generate and display report
    report = debugger.generate_report(results)
    print("\n" + report)
    
    # Save report if requested
    if args.report:
        Path(args.report).write_text(report)
        print(f"\n📄 Report saved to {args.report}")
    
    # Auto-commit if requested and fixes were applied
    if args.commit and results['fixes_applied']:
        commit_msg = f"Auto-fix: {', '.join(results['fixes_applied'])}"
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", commit_msg])
        print(f"\n✅ Committed fixes: {commit_msg}")
    
    # Exit with error code if issues remain
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
