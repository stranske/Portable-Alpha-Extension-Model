#!/usr/bin/env python3
"""
Minimal test script to check basic functionality in GitHub Actions
"""

import subprocess
import sys


def main():
    print("🧪 Minimal test starting...")

    # Test 1: Python basics
    print(f"✅ Python version: {sys.version}")

    # Test 2: Can we run gh command?
    try:
        result = subprocess.run(
            ["gh", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"✅ GitHub CLI version: {result.stdout.strip()}")
        else:
            print(f"❌ GitHub CLI failed: {result.stderr}")
            return 1
    except Exception as e:
        print(f"❌ GitHub CLI exception: {e}")
        return 1

    # Test 3: Can we check auth?
    try:
        result = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ GitHub CLI authenticated")
        else:
            print(f"⚠️  GitHub CLI auth: {result.stderr}")
    except Exception as e:
        print(f"❌ Auth check exception: {e}")
        return 1

    # Test 4: Simple git command
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print(f"✅ Current branch: {result.stdout.strip()}")
        else:
            print(f"❌ Git failed: {result.stderr}")
            return 1
    except Exception as e:
        print(f"❌ Git exception: {e}")
        return 1

    print("🎉 All minimal tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
