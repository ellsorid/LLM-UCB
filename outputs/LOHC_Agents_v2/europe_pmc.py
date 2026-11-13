"""Retrieve all Europe PMC results for a query using cursor pagination."""

import html
import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def search(query, *, page_size=100, timeout=20):
    if not 1 <= page_size <= 1000:
        raise ValueError("Europe PMC page size must be between 1 and 1000")
    cursor = "*"
    seen = {cursor}
    records = []
    pages = 0
    while True:
        params = {"query": query, "format": "json", "resultType": "core",
                  "pageSize": page_size, "cursorMark": cursor}
        request = Request(SEARCH_URL + "?" + urlencode(params),
                          headers={"User-Agent": "LOHC literature retrieval"})
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        pages += 1
        batch = result["resultList"]["result"]
        records.extend(batch)
        hit_count = int(result["hitCount"])
        if len(records) >= hit_count:
            return {"hit_count": hit_count, "pages": pages, "records": records}
        next_cursor = result.get("nextCursorMark")
        if not batch or not next_cursor or next_cursor in seen:
            raise RuntimeError(f"Europe PMC pagination ended after {len(records)} of {hit_count} records")
        seen.add(next_cursor)
        cursor = next_cursor


def normalize(item):
    doi = (item.get("doi") or "").strip().lower().removeprefix("https://doi.org/") or None
    source = item.get("source")
    identifier = item.get("id")
    pmid = item.get("pmid") or (identifier if source == "MED" else None)
    pmcid = item.get("pmcid") or (identifier if source == "PMC" else None)
    abstract = item.get("abstractText") or ""
    abstract = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", abstract)).split()) or None
    source_url = None
    if doi:
        source_url = f"https://doi.org/{doi}"
    elif pmcid:
        source_url = f"https://europepmc.org/articles/{pmcid}"
    elif source and identifier:
        source_url = f"https://europepmc.org/article/{source}/{identifier}"
    links = (item.get("fullTextUrlList") or {}).get("fullTextUrl", [])
    return {"doi": doi, "pmid": pmid, "pmcid": pmcid, "source": source, "record_id": identifier,
            "title": item.get("title"), "abstract": abstract, "source_url": source_url,
            "full_text_links": links}


def identity(record):
    for key in ("doi", "pmid", "pmcid"):
        if record.get(key):
            return key, record[key]
    if record.get("source") and record.get("record_id"):
        return record["source"], record["record_id"]
    return None
