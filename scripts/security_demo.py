#!/usr/bin/env python3
"""
Demonstration of the command injection vulnerability fix.

This script shows the difference between the vulnerable os.system() approach
and the secure subprocess.run() approach.

DO NOT run the vulnerable examples in production!
"""

import subprocess
import sys
from pathlib import Path


def demonstrate_vulnerability():
    """Show how the vulnerable code could be exploited."""
    print("=== COMMAND INJECTION VULNERABILITY DEMONSTRATION ===")
    print()
    
    print("1. VULNERABLE CODE (DO NOT USE):")
    print("   import os")
    print("   python_exe = '/tmp/malicious; rm -rf /tmp/test; echo pwned'")
    print("   get_pip = '/tmp/get-pip.py'") 
    print("   os.system(f'\"{python_exe}\" {get_pip}')")
    print()
    print("   This would execute: /tmp/malicious; rm -rf /tmp/test; echo pwned /tmp/get-pip.py")
    print("   The semicolon allows command injection!")
    print()
    
    print("2. SECURE CODE (RECOMMENDED):")
    print("   import subprocess")
    print("   python_exe = '/tmp/malicious; rm -rf /tmp/test; echo pwned'")
    print("   get_pip = '/tmp/get-pip.py'")
    print("   subprocess.run([str(python_exe), str(get_pip)], check=True)")
    print()
    print("   This passes arguments as a list - no shell interpretation!")
    print("   The malicious characters are treated as literal filename characters.")
    print()


def demonstrate_secure_implementation():
    """Show the secure implementation with actual safe code."""
    print("=== SECURE IMPLEMENTATION EXAMPLE ===")
    print()
    
    # Safe example with harmless paths
    python_exe = Path("/usr/bin/python3")  # Standard path
    script_path = Path("/tmp/safe_script.py")
    
    # Create a safe test script
    script_path.write_text("print('Hello from secure subprocess!')\n")
    
    try:
        print(f"Executing: {python_exe} {script_path}")
        result = subprocess.run([str(python_exe), str(script_path)], 
                              capture_output=True, text=True, check=True)
        print(f"Output: {result.stdout.strip()}")
        print("✅ Secure execution successful!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    finally:
        # Clean up
        if script_path.exists():
            script_path.unlink()
    
    print()


def main():
    """Main demonstration function."""
    print("Command Injection Security Fix Demonstration")
    print("=" * 50)
    print()
    
    demonstrate_vulnerability()
    demonstrate_secure_implementation()
    
    print("KEY TAKEAWAYS:")
    print("• Never use os.system() with user-controlled input")
    print("• Always use subprocess.run() with argument lists")  
    print("• Use check=True for proper error handling")
    print("• Convert Path objects to strings explicitly")
    print("• This prevents shell injection attacks")
    print()


if __name__ == "__main__":
    main()