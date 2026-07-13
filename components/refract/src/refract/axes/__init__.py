"""Implemented REFRACT scoring axes.

- ``gtm``: deprecated text-retokenization compatibility axis.
- ``trajectory``: decode-time token-ID agreement.
- ``kld``: corpus-anchored distribution divergence.
- ``rniah``: long-context retrieval degradation.
- ``plad``: excess drift under small prompt perturbations.

See each module and ``components/refract/LIMITATIONS.md`` for protocols and backend
constraints.
"""

from __future__ import annotations
