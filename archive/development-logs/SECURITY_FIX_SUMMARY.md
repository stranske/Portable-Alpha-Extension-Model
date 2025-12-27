# Command Injection Security Fix - Summary

## Issue Description
PR #565 introduced Windows portable zip functionality in `scripts/make_portable_zip.py` that contained two `os.system()` calls vulnerable to command injection:

1. **Pip bootstrap**: `os.system(f'"{python_exe}" {get_pip}')`
2. **Dependency install**: `os.system(f'"{python_exe}" -m pip install -r "{req}" --no-warn-script-location')`

These calls format user-controlled paths into shell commands, allowing potential arbitrary code execution if paths contain malicious characters like `; rm -rf /`.

## Security Fix Applied

### Before (Vulnerable)
```python
# Vulnerable to command injection
os.system(f'"{staging / "python" / "python.exe"}" {get_pip}')
os.system(f'"{staging / "python" / "python.exe"}" -m pip install -r "{req}" --no-warn-script-location')
```

### After (Secure)  
```python
# Secure implementation using subprocess.run() with argument lists
subprocess.run([str(python_exe), str(get_pip)], check=True, cwd=staging)
subprocess.run([
    str(python_exe),
    "-m", "pip", "install", 
    "-r", str(req),
    "--no-warn-script-location"
], check=True, cwd=staging)
```

## Security Improvements

1. **No Shell Interpretation**: `subprocess.run()` with argument lists bypasses shell entirely
2. **Path Safety**: Malicious characters in paths are treated as literal filename characters
3. **Error Handling**: `check=True` ensures failures raise `CalledProcessError` exceptions
4. **Proper Exception Wrapping**: Subprocess errors are caught and wrapped in `RuntimeError`

## Testing

Created comprehensive security tests (`tests/test_make_portable_zip_security.py`):
- ✅ Verifies subprocess calls use argument lists (not shell strings)
- ✅ Confirms command injection is prevented with malicious paths  
- ✅ Tests proper error handling for subprocess failures
- ✅ Ensures no dangerous patterns remain in code

## Files Modified

- `scripts/make_portable_zip.py`: Applied security fix
- `tests/test_make_portable_zip_security.py`: Added security tests (new)
- `scripts/security_demo.py`: Added demonstration script (new)

## Validation

- All 4 security tests pass
- Basic functionality verified (archive creation works)
- Linting passes (ruff)
- No regression in existing functionality

This fix prevents command injection attacks while maintaining all intended functionality.