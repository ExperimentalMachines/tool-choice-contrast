"""tool-choice-contrast: a frozen ANSWER-behaviour evaluation corpus and its scorer."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# The vendored OpenGrad scorer imports its own siblings absolutely (`from opengrad.data.behavior
# import ...`), which is what keeps the files byte-identical to upstream. That only resolves if the
# vendor directory is on sys.path, so it is put there here rather than relying on a pytest setting
# that an installed console script does not see.
#
# Prepended, not appended: if a real `opengrad` is also installed, this package must still score
# with the pinned copy recorded in _vendor/opengrad/PROVENANCE.json. A benchmark whose numbers
# depend on what else happens to be in the environment is not a benchmark.
_VENDOR = str(_Path(__file__).resolve().parent / "_vendor")
if _VENDOR not in _sys.path:
    _sys.path.insert(0, _VENDOR)

from toolchoice.constants import CORPUS_ID as CORPUS_ID
from toolchoice.constants import SCHEMA_VERSION as SCHEMA_VERSION

__version__ = "0.1.0"
