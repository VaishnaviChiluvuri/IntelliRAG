# Backward-compatibility alias pointing to modern multi-source external knowledge
from backend.external_knowledge import (
    fetch_external_knowledge as fetch_wikipedia_fallback,
    extract_search_entity,
    fetch_wikipedia_knowledge,
    fetch_duckduckgo_knowledge,
)

__all__ = [
    "fetch_wikipedia_fallback",
    "extract_search_entity",
    "fetch_wikipedia_knowledge",
    "fetch_duckduckgo_knowledge",
]
