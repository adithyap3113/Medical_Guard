"""Values that must never change once data exists.

Everything else in this project is configuration: a setting can be tuned, an
environment variable can be overridden, and the worst outcome is a restart. The
constants here are different in kind. They are baked into identifiers that get
written into a vector store, and changing one silently invalidates every identifier
derived from it -- without any error, any warning, or any failing test at the moment
of the change. The damage surfaces later, as duplicate points and a corpus that
re-ingestion can no longer deduplicate.

So they live in their own module, with the reasoning attached, rather than in
``config.py`` where they would look tunable.
"""

from __future__ import annotations

import uuid

__all__ = ["MEDGUARD_UUID_NAMESPACE"]

#: Deterministic namespace for ``uuid.uuid5`` point-ID generation in Qdrant.
#:
#: Qdrant point ids must be an unsigned integer or a UUID -- an arbitrary string such
#: as ``"cdc_anthrax_2023::0007"`` is rejected. Phase 3B therefore bridges the chunk id
#: to a point id with ``uuid.uuid5(MEDGUARD_UUID_NAMESPACE, chunk_id)``, which is
#: deterministic: the same chunk id always maps to the same point, so re-ingesting an
#: unchanged document upserts over itself instead of duplicating.
#:
#: That determinism is entirely relative to this namespace. Change it and every id
#: changes with it: the old points stay in the collection, the new ones land beside
#: them, and idempotency is gone for every document already ingested. Nothing raises.
#:
#: Do NOT regenerate it, do NOT move it to config, do NOT read it from the environment
#: -- an environment-dependent namespace makes point ids differ between two machines
#: pointed at the same collection. If it genuinely has to be rotated, drop and rebuild
#: every collection as part of the same change.
#:
#: Generated once with ``python -c "import uuid; print(uuid.uuid4())"`` and committed as
#: a literal. ``tests/unit/test_uuid_namespace.py`` fails if this line is edited.
MEDGUARD_UUID_NAMESPACE: uuid.UUID = uuid.UUID("759d085e-6dcc-4956-bb53-900708f531ef")
