"""
pytest runs with --import-mode=importlib (GATAC's pyproject.toml), which leaves
test/ off sys.path; put it back so the tests import _timing as they do when run
as scripts.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
