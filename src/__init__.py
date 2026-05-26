"""
src — Gut Microbiome CNN source package.

Adds the project root to sys.path on first import so that all modules in this
package can do `from config import ...` without any extra path manipulation.
"""
import sys
import os

# Insert project root (one level above this file) so 'config' is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
