"""Create a portable Windows zip archive of the project.

This utility creates a clean, runtime-only distribution by filtering out
development artifacts and only including necessary files for deployment.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Set


def get_default_excludes() -> Set[str]:
    """Return default patterns to exclude from the portable archive.
    
    These patterns are based on .gitignore and common development artifacts
    that should not be included in a runtime distribution.
    """
    return {
        # Version control
        ".git",
        ".gitignore", 
        ".gitattributes",
        ".github",
        
        # Python caches and builds
        "__pycache__",
        "*.pyc",
        "*.pyo", 
        "*.egg-info",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        
        # Virtual environments
        ".venv",
        "venv",
        "dev-env",
        
        # Development tools and configs
        ".pre-commit-config.yaml",
        ".vscode",
        ".idea", 
        "*.swp",
        "*.swo",
        ".ipynb_checkpoints",
        ".jupyter",
        "pyrightconfig.json",
        
        # Development documentation and logs
        "DEVELOPMENT_*.md",
        "CODEX_*.md", 
        "TUTORIAL_*_TESTING_RESULTS.md",
        "AUTOMATION_QUICK_START.md",
        "COMPLETE_TUTORIAL_TESTING_SUMMARY.md",
        "DEMO_TESTING_LOG.md",
        "FUNCTIONALITY_GAP_ANALYSIS.md",
        "PARAMETER_SWEEP_STATUS_REPORT.md",
        "STREAMLIT_TUTORIAL_INTEGRATION_PLAN.md",
        "TUTORIAL_IMPLEMENTATION_DEPENDENCY.md",
        "TUTORIAL_UPDATE_DRAFT.md",
        "TEST_PERMISSIONS.md",
        "code-quality.log",
        "debug_*.md",
        "debug_*.py",
        "streamlined_*.md",
        "streamlined-*.yml",
        "test_debug_*.md",
        "tutorial*_issues.md",
        "user_testing_issues.md",
        "codex.patch",
        
        # Build artifacts and outputs
        "docs/_build",
        "plots",
        "*.xlsx",
        "*.tmp",
        "*.temp",
        ".env",
        ".env.local",
        "Outputs.parquet",
        "get-pip.py",
        
        # OS files
        ".DS_Store",
        "Thumbs.db",
        
        # Development containers and tools
        ".devcontainer",
        ".gate_smoke",
        "Makefile",
        "archive",
        
        # Jupyter notebooks (development)
        "*.ipynb",
    }


def get_default_includes() -> Set[str]:
    """Return patterns that should always be included in the portable archive.
    
    These are essential runtime files that the application needs to function.
    """
    return {
        # Core Python package
        "pa_core/**/*.py",
        
        # Dashboard and web interface
        "dashboard/**/*.py",
        
        # Configuration and templates
        "config/**/*",
        "templates/**/*",
        "config_theme.yaml",
        "config_thresholds.yaml",
        
        # Setup and requirements
        "setup.py",
        "pyproject.toml", 
        "requirements.txt",
        "requirements-dev.txt",
        
        # Sample configurations and data
        "my_first_scenario.yml",
        "sp500tr_fred_divyield.csv",
        "test_params.yml",
        "tutorial*.yml", 
        
        # Documentation
        "README.md",
        "docs/**/*.md",
        "docs/**/*.rst",
        "tutorials/**/*.md",
        
        # Scripts
        "scripts/launch_dashboard.bat",
        "scripts/launch_dashboard.sh",
        "dev.sh",
        "setup.sh",
        "setup_deps.sh",
        "activate_env.sh",
    }


def should_exclude_path(path: Path, root: Path, excludes: Set[str]) -> bool:
    """Check if a path should be excluded from the archive."""
    # Get relative path from root
    try:
        rel_path = path.relative_to(root)
        path_parts = rel_path.parts
        path_str = str(rel_path)
    except ValueError:
        return True  # Path is outside root, exclude it
    
    # Always exclude empty files (might be test artifacts)
    # Exclude empty Python files, except for empty __init__.py files (which are needed for package structure)
    if path.is_file() and path.stat().st_size == 0 and path.suffix == '.py' and path.name != '__init__.py':
        return True
        
    # Check against exclude patterns
    for exclude in excludes:
        # Direct name match
        if path.name == exclude or path_str == exclude:
            return True
            
        # Wildcard patterns
        if "*" in exclude:
            import fnmatch
            if fnmatch.fnmatch(path.name, exclude) or fnmatch.fnmatch(path_str, exclude):
                return True
                
        # Directory patterns (check if any parent directory matches)
        for part in path_parts:
            if part == exclude:
                return True
                
        # Prefix patterns for development docs
        if exclude.endswith("*") and path.name.startswith(exclude[:-1]):
            return True
            
    return False


def create_filtered_zip(root_dir: Path, output_path: Path, excludes: Set[str]) -> None:
    """Create a zip file with filtered content."""
    print(f"Creating portable archive: {output_path}")
    print(f"Source directory: {root_dir}")
    
    files_added = 0
    files_excluded = 0
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in root_dir.rglob('*'):
            if path.is_file():
                if should_exclude_path(path, root_dir, excludes):
                    files_excluded += 1
                    print(f"  EXCLUDED: {path.relative_to(root_dir)}")
                else:
                    arcname = path.relative_to(root_dir)
                    zipf.write(path, arcname)
                    files_added += 1
                    
    print(f"\nArchive created successfully!")
    print(f"Files included: {files_added}")
    print(f"Files excluded: {files_excluded}")
    print(f"Archive size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main() -> None:
    """Main entry point for the portable zip creation utility."""
    parser = argparse.ArgumentParser(
        description="Create a portable zip archive with development artifacts filtered out",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python make_portable_zip.py
  python make_portable_zip.py --output my_distribution.zip
  python make_portable_zip.py --verbose
        """
    )
    parser.add_argument(
        "--output", 
        default="portable_windows.zip", 
        help="Output zip file name (default: %(default)s)"
    )
    parser.add_argument(
        "--exclude-pattern", 
        action="append",
        default=[],
        help="Additional patterns to exclude (can be used multiple times)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Show detailed output including excluded files"
    )
    
    args = parser.parse_args()

    # Determine project root (parent of scripts directory)
    root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output).resolve()
    
    # Get exclude patterns
    excludes = get_default_excludes()
    
    # Add user-specified patterns  
    for pattern in args.exclude_pattern:
        excludes.add(pattern)
        
    if args.verbose:
        print(f"Total exclude patterns: {len(excludes)}")
    
    if not args.verbose:
        # Use a quieter version for non-verbose output
        def quiet_create_filtered_zip(root_dir: Path, output_path: Path, excludes: Set[str]) -> None:
            """Create zip with minimal output."""
            files_added = 0
            files_excluded = 0
            
            print(f"Creating portable archive: {output_path}")
            print(f"Source directory: {root_dir}")
            print("Filtering files...")
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for path in root_dir.rglob('*'):
                    if path.is_file():
                        if should_exclude_path(path, root_dir, excludes):
                            files_excluded += 1
                        else:
                            arcname = path.relative_to(root_dir)
                            zipf.write(path, arcname)
                            files_added += 1
                            
            print(f"\nArchive created successfully!")
            print(f"Files included: {files_added}")
            print(f"Files excluded: {files_excluded}")
            print(f"Archive size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        create_filtered_zip_func = quiet_create_filtered_zip
    else:
        create_filtered_zip_func = create_filtered_zip
    
    # Create the filtered archive
    try:
        create_filtered_zip_func(root, output_path, excludes)
    except Exception as e:
        print(f"Error creating archive: {e}")
        return 1
        
    return 0


if __name__ == "__main__":  # pragma: no cover - utility script
    exit(main())