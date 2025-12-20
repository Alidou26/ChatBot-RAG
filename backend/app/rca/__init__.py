"""Root cause analysis package.

Root cause analysis (RCA) attempts to identify the fundamental reason
behind a problem rather than merely addressing its symptoms. In the
context of a knowledge management assistant this involves searching
through the available technical documentation to find relevant
information that explains why a given issue occurs and summarising
those findings.

This package exposes the :func:`analyse_root_cause` function which
ties together retrieval and answer generation to provide an
actionable diagnosis.
"""

from .analysis import analyse_root_cause

__all__ = ["analyse_root_cause"]