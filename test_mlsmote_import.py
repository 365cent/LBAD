#!/usr/bin/env python3
"""
Quick test to check if MLSMOTE can be imported.
Run this with: python test_mlsmote_import.py
"""

import sys

print("=" * 60)
print("Testing MLSMOTE Import")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}\n")

# Test the exact import logic from transformer.py
MLSMOTE_AVAILABLE = False

try:
    from skmultilearn.adapt import MLSMOTE
    MLSMOTE_AVAILABLE = True
    print("✓ SUCCESS: Imported from skmultilearn.adapt")
    print(f"  MLSMOTE class: {MLSMOTE}")
except ImportError as e:
    print(f"✗ FAILED: skmultilearn.adapt.MLSMOTE - {e}")
    try:
        # Try alternative import path
        from skmultilearn.problem_transform import MLSMOTE
        MLSMOTE_AVAILABLE = True
        print("✓ SUCCESS: Imported from skmultilearn.problem_transform")
        print(f"  MLSMOTE class: {MLSMOTE}")
    except ImportError as e2:
        print(f"✗ FAILED: skmultilearn.problem_transform.MLSMOTE - {e2}")
        MLSMOTE_AVAILABLE = False

print("\n" + "=" * 60)
print("Result")
print("=" * 60)

if MLSMOTE_AVAILABLE:
    print("✓ MLSMOTE is available and can be used")
    print("\nYour transformer.py will use the actual MLSMOTE library.")
else:
    print("✗ MLSMOTE is NOT available")
    print("\nYour transformer.py will use the fallback k-NN SMOTE method.")
    print("This is fine - the fallback method works well and balances all classes.")
    
print("\n" + "=" * 60)
print("Checking scikit-multilearn installation")
print("=" * 60)

try:
    import skmultilearn
    print(f"✓ scikit-multilearn installed")
    
    # Try to get version, but don't fail if it doesn't exist
    try:
        print(f"  Version: {skmultilearn.__version__}")
    except AttributeError:
        print(f"  Version: (version info not available)")
    
    print(f"  Location: {skmultilearn.__file__}")
    
    # Check what's available in skmultilearn
    print("\n  Available submodules:")
    import pkgutil
    for importer, modname, ispkg in pkgutil.iter_modules(skmultilearn.__path__):
        print(f"    - {modname}")
    
    # Check specifically for adapt and problem_transform
    print("\n  Checking adapt module:")
    try:
        from skmultilearn import adapt
        print(f"    Contents: {[x for x in dir(adapt) if not x.startswith('_')]}")
    except ImportError:
        print("    ✗ adapt module not found")
    
    print("\n  Checking problem_transform module:")
    try:
        from skmultilearn import problem_transform
        print(f"    Contents: {[x for x in dir(problem_transform) if not x.startswith('_')]}")
    except ImportError:
        print("    ✗ problem_transform module not found")
        
except ImportError:
    print("✗ scikit-multilearn is NOT installed")
    print("\n  Install with:")
    print("    pip install scikit-multilearn==0.2.0")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("The transformer.py script will work either way:")
print("  - If MLSMOTE available: Uses library implementation (slower but more accurate)")
print("  - If MLSMOTE not available: Uses k-NN fallback (faster, still good quality)")
print("\nBoth methods balance all classes (including normal) to equal size.")

