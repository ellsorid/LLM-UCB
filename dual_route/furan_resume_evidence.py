"""Scoped primary-literature analogues for mapped furan/THF units."""


PRIMARY = (
    ("10.1016/j.jcat.2015.04.018", "https://www.sciencedirect.com/science/article/abs/pii/S0021951715001128",
     "lean","Supported noble metal study; see publication for run conditions",
     "The study experimentally converted furan to tetrahydrofuran, retaining the five-membered oxygen ring."),
    ("10.1021/acssuschemeng.4c08647", "https://doi.org/10.1021/acssuschemeng.4c08647",
     "rich","Tetrahydrofuran dehydrogenation; source compares catalysts and conditions",
     "In experiments on tetrahydrofuran, furan was the main product of dehydrogenation."),
    ("10.1039/C8NJ05152F", "https://pubs.rsc.org/en/content/articlehtml/2019/nj/c8nj05152f",
     "lean","Furan hydrogenation experiment; ring-opening selectivity is condition-dependent",
     "The furan hydrogenation study identified further ring opening as a competing reaction under some conditions."),
)


def scoped_furan_analogue(pair, units):
    by_state = {}
    for state in ("lean", "rich"):
        blocks = units["units"]["ring_or_conjugation"]
        by_state[state] = {tuple(b["parent_ids"]):b for b in blocks
                           if b["state"]==state and b["name"]=="ring_system"}
    hits, payload = [], []
    for ids in sorted(set(by_state["lean"]) & set(by_state["rich"])):
        lean,rich = (by_state[state][ids] for state in ("lean","rich"))
        if (len(ids)!=5 or sorted(x[1] for x in lean["atom_labels"])!=[6,6,6,6,8]
            or len(lean["internal_bonds"])!=5 or len(rich["internal_bonds"])!=5
            or not all(x[4] for x in lean["atom_labels"])
            or not all(x[2]==1 for x in rich["internal_bonds"])):
            continue
        substituted = {i for i,_,_ in lean["boundary_bonds"]}
        # Exact paired ring core; reduce context match for each substituted site.
        similarity = (len(ids)-len(substituted))/len(ids)
        for doi,url,state,catalyst,observation in PRIMARY:
            ref = "primary:"+doi
            obs_id = f"F{len(hits)+1}"
            hits.append({"id":obs_id,"source_id":doi,"reference":ref,"url":url,"state":state,
                         "unit":"mapped_furan_THF","target_ids":list(ids),"similarity":similarity,
                         "source_catalyst":catalyst,"used_for_judgement":True,
                         "transfer_scope":"unsubstituted ring analogue; no target or edit-specific measurement"})
            payload.append({"id":obs_id,"unit_type":"mapped furan/THF ring analogue","parent_ids":list(ids),
                            "state":state,"observation":observation})
        # Both target rings share the same external observations. Score each DOI once.
        break
    return hits,payload
