"""Retrieve platform literature and qualify only exact, source-verified biomass routes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import sys

from rdkit import Chem

from ac_qd import HERE, agents, atomic_json, evidence_records, read_json, usage

sys.path.insert(0, str(HERE.parent / "LOHC_Agents_v2"))
import europe_pmc

PLATFORMS = {
    "phenol": ("phenol", "PHENOL-PINE-ROUTE"),
    "guaiacol_like": ("4-propylguaiacol", "PROPYLGUAIACOL-BIRCH"),
}
PROMPT = """核对生物基路线。目标名称与SMILES由代码提供。摘要明确报道该目标在特定生物质的转化产物中（包括明确富含该目标的混合产物）即可标为supported，不能假设已分离纯物。给出含目标名称的逐字原文 quote，以及原文中的生物质 feedstock 和转化 conversion。前体和产物允许分别出现在同一摘要不同句子。只返回 JSON 对象 {"status":"unknown","feedstock":null,"conversion":null,"quote":null}。supported时后三项为字符串。"""


def norm(value):
    return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", value.lower()).split())


def get_route(platform, info, evidence, api, output):
    name, evidence_id = info
    reference = evidence[evidence_id]
    doi = reference["doi"].lower()
    query = f'DOI:"{doi}" AND TITLE_ABS:(lignin OR biomass OR birch OR pinewood)'
    try:
        found = europe_pmc.search(query, timeout=12)
        records = [europe_pmc.normalize(r) for r in found["records"]]
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {"status": "retrieval_failed", "query": query, "error": type(exc).__name__}
    paper = next((r for r in records if r["doi"] == doi and r["abstract"]), None)
    audit = {"query": query, "retrieved": len(records), "source_doi": doi, "status": "unknown"}
    if not paper:
        return audit
    packet = {"platform": platform, "target_name": name, "target_smiles": reference["lean_smiles"],
              "title": paper["title"], "abstract": paper["abstract"], "doi": doi}
    atomic_json(output / f"{platform}_input.json", packet)
    try:
        reply = api.run(platform, PROMPT, packet)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        return {**audit, "status": "extraction_failed", "error": type(exc).__name__}
    quote = reply.get("quote")
    feedstock = reply.get("feedstock")
    conversion = reply.get("conversion")
    if (set(reply) == {"status", "feedstock", "conversion", "quote"} and reply["status"] == "supported"
            and all(isinstance(item, str) and item.strip() for item in (quote, feedstock, conversion))
            and norm(quote) in norm(paper["abstract"]) and norm(name) in norm(quote)
            and norm(feedstock).replace(" ", "") in norm(paper["abstract"]).replace(" ", "")):
        audit.update(status="published_exact_route", lean_smiles=Chem.MolToSmiles(
            Chem.MolFromSmiles(reference["lean_smiles"])), feedstock=feedstock, conversion=conversion, quote=quote)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--output", default="provenance")
    args = parser.parse_args()
    output = args.run / args.output
    output.mkdir()
    evidence = {row["id"]: row for row in evidence_records()}
    api = agents.DeepSeek(agents.credential(args.key_file), output)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {p: pool.submit(get_route, p, cfg, evidence, api, output) for p, cfg in PLATFORMS.items()}
        routes = {p: result.result() for p, result in pending.items()}
    atomic_json(output / "routes.json", {"platforms": routes, "usage": usage(output)})
    print({p: r["status"] for p, r in routes.items()}, flush=True)


if __name__ == "__main__":
    main()
