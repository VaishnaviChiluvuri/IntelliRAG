import re
import urllib.parse
import logging
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger("intellirag.external_knowledge")

WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
USER_AGENT = "IntelliRAG-MultiHop/2.0 (intellirag.research@gmail.com) httpx/0.28.1"


def extract_search_entity(query: str) -> str:
    """
    Extracts the core entity or topic from a natural language question.
    e.g. 'What is the process of condensation?' -> 'condensation'
    """
    q = query.strip().rstrip("?.!")
    cleaned = re.sub(
        r"^(?:what\s+is\s+|what\s+are\s+|who\s+is\s+|who\s+was\s+|define\s+|explain\s+|describe\s+|how\s+does\s+|why\s+does\s+|which\s+is\s+)",
        "",
        q,
        flags=re.I,
    ).strip()

    cleaned = re.sub(
        r"^(?:the\s+process\s+of|the\s+definition\s+of|the\s+concept\s+of|the\s+difference\s+between|the\s+term)\s+",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    cleaned = re.sub(r"\s+(?:called|known as|defined as|refer to)$", "", cleaned, flags=re.I).strip()

    return cleaned if len(cleaned) >= 3 else q


def fetch_duckduckgo_knowledge(query: str) -> Dict[str, Any]:
    """Queries DuckDuckGo Instant Answer API for quick definitions and technical topics."""
    entity = extract_search_entity(query)
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(entity)}&format=json&no_html=1&skip_disambig=1"
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("AbstractText") or data.get("Abstract") or ""
            source = data.get("AbstractSource") or "DuckDuckGo Instant Answers"
            source_url = data.get("AbstractURL") or "https://duckduckgo.com"
            heading = data.get("Heading") or entity

            if abstract and len(abstract.strip()) > 30:
                return {
                    "used": True,
                    "source": f"External Knowledge ({source})",
                    "title": heading,
                    "url": source_url,
                    "summary": abstract.strip(),
                    "query": query,
                }
    except Exception as exc:
        logger.debug(f"DuckDuckGo search error: {exc}")
    return {"used": False}


def fetch_wikipedia_knowledge(query: str) -> Dict[str, Any]:
    """Queries Wikipedia API with clean entity extraction and disambiguation avoidance."""
    entity = extract_search_entity(query)
    if not entity:
        return {"used": False, "message": "Could not determine topic for external search."}

    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(timeout=5.0, follow_redirects=True, headers=headers) as client:
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": entity,
                "srlimit": 3,
                "format": "json",
            }
            resp = client.get(WIKI_SEARCH_URL, params=search_params)
            if resp.status_code != 200:
                return {"used": False, "message": f"Wikipedia returned HTTP {resp.status_code}"}

            hits = resp.json().get("query", {}).get("search", [])
            if not hits:
                return {"used": False, "message": f"No articles found for '{entity}'"}

            selected_title = None
            for hit in hits:
                title = hit.get("title", "")
                snippet = hit.get("snippet", "").lower()
                if "may refer to:" in snippet or "disambiguation" in title.lower():
                    continue
                selected_title = title
                break

            if not selected_title:
                selected_title = hits[0].get("title")

            encoded_title = urllib.parse.quote(selected_title.replace(" ", "_"))
            summary_resp = client.get(f"{WIKI_SUMMARY_URL}/{encoded_title}")
            if summary_resp.status_code != 200:
                return {"used": False, "message": f"Failed fetching summary for '{selected_title}'"}

            sdata = summary_resp.json()
            extract = sdata.get("extract", "").strip()
            page_url = (
                sdata.get("content_urls", {})
                .get("desktop", {})
                .get("page", f"https://en.wikipedia.org/wiki/{encoded_title}")
            )

            if not extract or len(extract) < 25:
                return {"used": False, "message": "Wikipedia summary was empty"}

            return {
                "used": True,
                "source": "External Knowledge (Wikipedia)",
                "title": selected_title,
                "url": page_url,
                "summary": extract[:750].strip(),
                "query": query,
            }
    except Exception as exc:
        logger.warning(f"Wikipedia fetch error: {exc}")
        return {"used": False, "message": str(exc)}


def fetch_external_knowledge(query: str) -> Dict[str, Any]:
    """
    Multi-source external knowledge retriever:
    Tries Wikipedia first for comprehensive encyclopedia context;
    falls back to DuckDuckGo Instant Answers.
    """
    wiki_res = fetch_wikipedia_knowledge(query)
    if wiki_res.get("used"):
        return wiki_res

    ddg_res = fetch_duckduckgo_knowledge(query)
    if ddg_res.get("used"):
        return ddg_res

    return {
        "used": False,
        "message": f"No external verified knowledge found for: '{query}'",
        "query": query,
    }

