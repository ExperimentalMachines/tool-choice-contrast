"""Namespace package only.

Upstream's ``src/opengrad/data/__init__.py`` re-exports ``adapters``,
``materialize``, ``normalize`` and ``renderers``. None of those are vendored
here -- they pull in the corpus-construction and training subsystems, which a
benchmark package has no business carrying. Every vendored module imports its
siblings absolutely (``from opengrad.data.behavior import ...``), so an empty
package init is sufficient and nothing upstream is shadowed.

This file is the ONLY deviation from byte-identity in the vendor drop, and it is
recorded as such in ``third_party/opengrad/PROVENANCE.json`` under
``replaced_files``.
"""
