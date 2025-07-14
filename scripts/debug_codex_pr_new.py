#!/usr/bin/env python3
"""
Methodical debugging workflow for Codex Pull Requests.
Focuses on specific CI/CD errors rather than shotgun debugging.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class CodexPRDebugger:
    """Methodical debugging workflow focusing on actual CI/CD failures."""
    
    def __init__(self, branch_name: Optional[str] = None):
        self.branch_name = branch_name
        self.repo_root = Path.cwd()
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
    
    def get_specific_ci_errors(self) -> Dict[str, List[str]]:
        """Get specific errors from CI/CD tools that are actually failing."""
        errors = {
            'mypy': [],
            'flake8': [],
            'pytest': [],
            'ruff': []
        }
        
        # Check mypy (Type Checking) - exact CI/CD command
        print("🔍 Checking mypy (Type Checking)...")
        success, output = self.run_command("python -m mypy pa_core/ --strict")
        if not success:
            for line in output.split('\n'):
                if 'error:' in line and line.strip():
                    errors['mypy'].append(line.strip())
        
        # Check flake8 (Code Quality) - exact CI/CD command  
        print("🔍 Checking flake8 (Code Quality)...")
        success, output = self.run_command(
            "flake8 pa_core/ tests/ dashboard/ --max-line-length=88 --ignore=E203,W503"
        )
        if not success:
            for line in output.split('\n'):
                if line.strip() and ':' in line:
                    errors['flake8'].append(line.strip())
        
        # Check pytest (Tests) - exact CI/CD command
        print("🔍 Checking pytest (Tests)...")
        success, output = self.run_command("python -m pytest tests/ -v --tb=short")
        if not success:
            for line in output.split('\n'):
                if 'FAILED' in line or 'ERROR' in line:
                    errors['pytest'].append(line.strip())
        
        # Check ruff formatting - exact CI/CD command
        print("🔍 Checking ruff formatting...")
        success, output = self.run_command("ruff format --check pa_core/ tests/ dashboard/")
        if not success:
            for line in output.split('\n'):
                if line.strip():
                    errors['ruff'].append(line.strip())
        
        # Filter out empty error lists
        return {k: v for k, v in errors.items() if v}
    
    def fix_mypy_errors(self, errors: List[str]) -> bool:
        """Fix specific mypy type annotation errors."""
        print("🔧 Fixing mypy type annotation errors...")
        
        fixed_any = False
        
        for error in errors:
            print(f"  📝 Found mypy error: {error}")
            
            if "Function is missing a type annotation" in error:
                # Extract file and line number for manual review
                if "pa_core/" in error:
                    file_path = error.split(":")[0]
                    print(f"    ⚠️  Manual fix needed in {file_path}")
            
            elif "Missing type parameters for generic type" in error:
                print("    ⚠️  Generic type parameter issue - may need manual fix")
            
            elif "Returning Any from function" in error:
                print("    ⚠️  Return type annotation issue - may need manual fix")
        
        return fixed_any
    
    def fix_flake8_errors(self, errors: List[str]) -> bool:
        """Fix specific flake8 style errors."""
        print("🔧 Fixing flake8 style errors...")
        
        fixed_any = False
        
        for error in errors:
            print(f"  📝 Found flake8 error: {error}")
            
            if "line too long" in error:
                print("    🔧 Applying black formatting to fix line length...")
                success, _ = self.run_command("black pa_core/ tests/ dashboard/")
                if success:
                    fixed_any = True
                    
            elif "imported but unused" in error:
                print("    ⚠️  Unused import - manual review recommended")
                
        if fixed_any:
            self.fixes_applied.append("Applied black formatting to fix line length issues")
            
        return fixed_any
    
    def fix_pytest_errors(self, errors: List[str]) -> bool:
        """Analyze pytest errors (usually require manual intervention)."""
        print("🔧 Analyzing pytest errors...")
        
        for error in errors:
            print(f"  ❌ Test failure: {error}")
            print("    ⚠️  Test failures typically require manual fixes")
            
        # Pytest failures usually require manual fixes
        return False
    
    def fix_ruff_errors(self, errors: List[str]) -> bool:
        """Fix ruff formatting errors."""
        print("🔧 Fixing ruff formatting errors...")
        
        for error in errors:
            print(f"  📝 Found ruff error: {error}")
        
        # Apply ruff formatting
        success, _ = self.run_command("ruff format pa_core/ tests/ dashboard/")
        if success:
            self.fixes_applied.append("Applied ruff formatting fixes")
            return True
            
        return False
    
    def run_methodical_debugging(self, max_iterations: int = 3) -> Dict[str, any]:
        """Run methodical debugging focusing on actual CI/CD failures."""
        print("🎯 Starting methodical Codex PR debugging...")
        print("📋 Focusing on specific CI/CD failures, not shotgun debugging")
        
        iteration = 0
        all_fixes_applied = []
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            
            # Get specific errors from CI/CD tools
            errors = self.get_specific_ci_errors()
            
            if not errors:
                print("✅ No CI/CD errors found!")
                break
            
            print(f"🔧 Found errors in {len(errors)} tools: {list(errors.keys())}")
            
            # Reset fixes for this iteration
            self.fixes_applied = []
            
            # Apply targeted fixes for each tool
            if 'mypy' in errors:
                self.fix_mypy_errors(errors['mypy'])
            
            if 'flake8' in errors:
                self.fix_flake8_errors(errors['flake8'])
            
            if 'pytest' in errors:
                self.fix_pytest_errors(errors['pytest'])
            
            if 'ruff' in errors:
                self.fix_ruff_errors(errors['ruff'])
            
            # Collect fixes
            if self.fixes_applied:
                all_fixes_applied.extend(self.fixes_applied)
                print(f"🔧 Applied {len(self.fixes_applied)} fixes this iteration")
            else:
                print("⚠️ No automatic fixes available - manual intervention needed")
                break
        
        # Final validation
        print("\n🔄 Final validation...")
        final_errors = self.get_specific_ci_errors()
        
        return {
            'iterations': iteration,
            'total_fixes_applied': all_fixes_applied,
            'final_errors': final_errors,
            'success': len(final_errors) == 0,
            'ci_cd_ready': len(final_errors) == 0
        }
    
    def generate_report(self, results: Dict[str, any]) -> str:
        """Generate a concise debugging report."""
        report = []
        
        report.append("# 🎯 Methodical Codex PR Debugging Report")
        report.append(f"Branch: {self.branch_name}")
        report.append(f"Iterations: {results['iterations']}")
        report.append("")
        
        if results['total_fixes_applied']:
            report.append("## ✅ Fixes Applied")
            for fix in results['total_fixes_applied']:
                report.append(f"- {fix}")
            report.append("")
        
        if results['final_errors']:
            report.append("## ❌ Remaining CI/CD Issues")
            for tool, errors in results['final_errors'].items():
                report.append(f"### {tool.upper()}")
                for error in errors[:5]:  # Limit to first 5 errors per tool
                    report.append(f"- {error}")
                if len(errors) > 5:
                    report.append(f"- ... and {len(errors) - 5} more errors")
                report.append("")
        
        # Status summary
        if results['ci_cd_ready']:
            report.append("## 🎉 Status: CI/CD READY")
            report.append("✅ All checks pass. Ready for GitHub Actions.")
        else:
            report.append("## ❌ Status: MANUAL FIXES NEEDED")
            report.append("⚠️ Some issues require manual intervention.")
            report.append("")
            report.append("### 📋 Recommended Actions:")
            if 'mypy' in results['final_errors']:
                report.append("1. Fix type annotations in reported functions")
            if 'pytest' in results['final_errors']:
                report.append("2. Review and fix failing tests")
            if 'flake8' in results['final_errors']:
                report.append("3. Review and fix remaining style issues")
            if 'ruff' in results['final_errors']:
                report.append("4. Apply additional formatting fixes")
        
        return "\n".join(report)


def main():
    """Main entry point for methodical debugging."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Methodical Codex PR Debugging")
    parser.add_argument("--branch", help="Branch name being debugged")
    parser.add_argument("--report", help="Output file for debugging report")
    parser.add_argument("--commit", action="store_true", help="Auto-commit fixes")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum debugging iterations")
    
    args = parser.parse_args()
    
    debugger = CodexPRDebugger(args.branch)
    
    # Run methodical debugging
    results = debugger.run_methodical_debugging(max_iterations=args.max_iterations)
    
    # Generate and display report
    report = debugger.generate_report(results)
    print("\n" + report)
    
    # Save report if requested
    if args.report:
        Path(args.report).write_text(report)
        print(f"\n📄 Report saved to {args.report}")
    
    # Auto-commit if requested and fixes were applied
    if args.commit and results['total_fixes_applied']:
        commit_msg = f"Methodical fix: {len(results['total_fixes_applied'])} targeted CI/CD fixes"
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", commit_msg])
        print(f"\n✅ Committed fixes: {commit_msg}")
    
    # Exit with appropriate status
    if results['ci_cd_ready']:
        print("\n🎉 Branch is CI/CD ready!")
        sys.exit(0)
    else:
        print(f"\n⚠️ Manual fixes needed for {len(results['final_errors'])} tools")
        sys.exit(1)


if __name__ == "__main__":
    main()
