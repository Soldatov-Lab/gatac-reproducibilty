"""
Where the tests keep their data: inputs they cannot download, everything they
write and the cached reference outputs. That is the directory in
$GATAC_REPRO_DATA (``run_suite.py --data-dir`` sets it), or data/ in this
repository.
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(os.environ.get("GATAC_REPRO_DATA") or os.path.join(REPO, "data"))


def data_path(*parts):
    """Path under the data directory."""
    return os.path.join(DATA, *parts)
