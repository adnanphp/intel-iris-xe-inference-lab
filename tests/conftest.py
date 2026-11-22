"""Add project root to sys.path so tests can import benchmark modules."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
