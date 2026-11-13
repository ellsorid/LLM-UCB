"""Single-pair matrix-to-search-to-memory-to-Critic pilot."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rdkit import Chem

from lohc_structure_score_trial import DeepSeek, credential, load, save
from lohc_three_matrices import encode
from lohc_matrix_protocol import DENSE, RULES
import europe_pmc


PAIR = ('Cn1ccc2ccccc21', 'CN1CCC2CCCCC21')
SCHEMA = RULES + '\n' + DENSE
SEARCH = ('Use the three matrices to infer the structural family, substituent position and change in hydrogenation. '
          'Propose two English literature search queries, one exact structural identity and one broader family/property query. '
          'Use keywords suitable for scholarly indexes. Do not supply a DOI or invent an observation. '
          'Return JSON {"structure":"short inferred structure", "queries":["exact", "broader"]}.')
EXTRACT = ('Extract ONLY the source facts explicitly supported by the supplied primary source. Return JSON '
           '{"observations":[{"id":"E1","source_id":"given id","scope":"reaction_enthalpy or reactive_system_decomposition",'
           '"applies_to":"P or system","direction":"rich_to_lean or both","value":55,"uncertainty":1,"unit":"kJ/mol H2",'
           '"finding":"concise paraphrase","anchor":"given anchor"}]}. Preserve sign and distinguish reaction direction. '
           'Use null for missing numerical fields. Reactive-system observations do not establish state-specific shelf stability. '
           'Confirm the named pair corresponds to the supplied matrices; do not extract a different isomer.')
CRITIC = ('Score only from the three matrices and admitted memory. Return JSON {"AR":{"score":null,"evidence_ids":[], '
          '"sites":[],"reason":"..."},"AL":{"score":null,"evidence_ids":[],"sites":[],"reason":"..."}, '
          '"P":{"score":0,"cycle_evidence":"unknown","evidence_ids":["E1"],"sites":[[0,1]],"reason":"..."},'
          '"readback":{"h2_wt_percent":0,"dehydrogenation_kj_per_mol_h2":0,"in_40_70_window":true},'
          '"unknowns":["..."]}. AR and AL are state-specific stability. Return null for unsupported state stability. '
          'For P use a 0–100 heuristic of how promising the documented thermochemistry and structural conversion are. '
          'Cite both relevant supported observations if applicable. P sites must be nonzero changed bonds in P. '
          'Identify thermal or reactive-system evidence as such, with unknown long-term cyclic durability and actual biomass origin. '
          'Do not extrapolate to ambient storage stability or report a comprehensive score when components are unknown.')


def request_json(url, params):
    request = Request(url + '?' + urlencode(params), headers={'User-Agent': 'LOHC-Matrix-Evidence-Pilot/1.0'})
    with urlopen(request, timeout=18) as response:
        return json.load(response)


def graph_identity(state):
    n = len(state)
    mol = Chem.RWMol()
    bonds = {1: Chem.BondType.SINGLE, 1.5: Chem.BondType.AROMATIC, 2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE}
    for row in state:
        z, charge, hydrogens, isotope, aromatic = map(int, row[n:])
        atom = Chem.Atom(z)
        atom.SetFormalCharge(charge)
        atom.SetNumExplicitHs(hydrogens)
        atom.SetNoImplicit(True)
        atom.SetIsotope(isotope)
        atom.SetIsAromatic(bool(aromatic))
        mol.AddAtom(atom)
    for i, row in enumerate(state):
        for j in range(i+1, n):
            if row[j]:
                mol.AddBond(i, j, bonds[row[j]])
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def search(out, api, assisted=False, matrices=None):
    matrices = encode(PAIR) if matrices is None else matrices
    save(out / 'query_matrices.json', matrices)
    packet = {'schema': SCHEMA, 'matrices': matrices}
    if assisted:
        identities = {}
        for k in ('AL','AR'):
            smiles = graph_identity(matrices[k])
            key = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
            request = Request(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{key}/property/IUPACName/JSON')
            try:
                with urlopen(request, timeout=12) as response:
                    name = json.load(response)['PropertyTable']['Properties'][0]['IUPACName']
            except (OSError, KeyError, IndexError):
                name = None
            identities[k] = {'smiles': smiles, 'inchikey': key, 'registry_iupac': name}
        packet['derived_structure_identifiers'] = identities
    save(out / 'search_agent_input.json', packet)
    plan = api.run('search_agent', SEARCH, packet)
    queries = plan['queries']
    if len(queries) != 2 or any(not isinstance(q, str) or not 5 <= len(q) <= 300 for q in queries):
        raise ValueError('Expected two search queries')
    save(out / 'search_plan.json', plan)
    if assisted and identities['AL']['registry_iupac']:
        name = identities['AL']['registry_iupac']
        queries = [*queries, f'"{name}" hydrogen storage equilibrium dehydrogenation enthalpy']
    records, log = {}, []
    for q in queries:
        for engine in ('Europe PMC', 'Crossref'):
            try:
                if engine == 'Europe PMC':
                    data = request_json(europe_pmc.SEARCH_URL, {'query': q, 'format': 'json', 'resultType': 'core', 'pageSize': 20})
                    hits = [europe_pmc.normalize(r) for r in data['resultList']['result']]
                    count = data['hitCount']
                else:
                    data = request_json('https://api.crossref.org/works', {'query.bibliographic': q, 'rows': 15})['message']
                    hits = [{'doi': r['DOI'].lower(), 'title': (r.get('title') or [''])[0], 'source_url': r.get('URL')}
                            for r in data['items'] if r.get('DOI')]
                    count = data['total-results']
                log.append({'engine': engine, 'query': q, 'total_hits': count, 'returned': len(hits),
                            'dois': [r['doi'] for r in hits if r.get('doi')]})
                for row in hits:
                    if row.get('doi'):
                        records.setdefault(row['doi'], row)
            except (OSError, KeyError, ValueError) as error:
                log.append({'engine': engine, 'query': q, 'error': str(error)})
    save(out / 'search_log.json', log)
    save(out / 'discovered.json', list(records.values()))
    print(f'Discovered {len(records)} DOI records', flush=True)


def matrix_pair(matrices):
    return tuple(graph_identity(matrices[k]) for k in ('AL', 'AR'))


def assess(out, api, sources, memory_only=False):
    query = load(out / 'query_matrices.json')
    pair = matrix_pair(query)
    discovered = {r['doi'] for r in load(out / 'discovered.json')}
    if any(r['doi'] not in discovered for r in sources):
        raise ValueError('At least one evidence source was not discovered by the search')
    packet = {'schema': SCHEMA, 'matrices': query,
              'sources': [{k: r[k] for k in ('source_id','doi','anchor','text')} for r in sources]}
    save(out / 'extract_input.json', packet)
    extracted = api.run('extract', EXTRACT, packet)['observations']
    by_id = {r['source_id']: r for r in sources}
    if len(extracted) != len(sources) or {r['source_id'] for r in extracted} != set(by_id):
        raise ValueError('Missing or duplicated source extraction')
    memory = []
    for row in extracted:
        source = by_id[row['source_id']]
        for field in ('scope', 'applies_to', 'direction', 'value', 'uncertainty', 'unit', 'anchor'):
            if row[field] != source[field]:
                raise ValueError(f'Extraction mismatch for {row["source_id"]}: {field}')
        memory.append({**row, 'doi': source['doi'], 'url': source['url'], 'conditions': source['conditions'],
                       'matrices': encode(pair, source['value']), 'source_kind': 'primary_paper'})
    if len({r['id'] for r in memory}) != len(memory):
        raise ValueError('Duplicate evidence ID')
    save(out / 'memory.json', memory)
    h = next(r['value'] for r in memory if r['scope'] == 'reaction_enthalpy')
    enriched = encode(pair, h)
    save(out / 'enriched_matrices.json', enriched)
    if memory_only:
        print(f'Admitted {len(memory)} verified observations', flush=True)
        return
    score(out, api, enriched, memory)


def score(out, api, enriched, memory):
    from lohc_matrix_protocol import BLOCKS
    from lohc_matrix_readback import cross_matrix, packet

    blocks = packet(enriched, 'critic', False)
    references = {r['id']: r for r in memory}
    observations = [{k: v for k, v in r.items() if k != 'matrices'} for r in memory]
    request = {'schema': RULES + '\n' + BLOCKS, 'matrices': blocks, 'memory': observations,
               'verified_bond_examples': cross_matrix(blocks)}
    save(out / 'critic_input.json', request)
    instruction = CRITIC + (' In verified_bond_examples each row is [lean_i,lean_j,rich_i,rich_j,lean_order,rich_order,delta]. '
                            'Cite P sites in lean coordinates and label literature-derived enthalpy and theoretical capacity. '
                            'For the P item include "cycle_evidence":"unknown" unless there are explicit long-term cycling data. '
                            'A structural mapping plus equilibrium thermochemistry supports a candidate transformation under the study '
                            'conditions. Do not assert full, complete, proven or experimentally confirmed reversibility. '
                            'When a state has no state-specific evidence, return an empty sites list for that state.')
    result = api.run('critic', instruction, request)
    n = len(enriched['AL'])
    p = enriched['P']
    h = p[-1][-2]
    sites = result['P']['sites']
    read = result['readback']
    reason = re.sub(r'\b(?:does not assert|does not demonstrate|no evidence of|cannot establish)\s+'
                    r'(?:fully reversible|proven reversibility|confirmed reversibility)\b', '',
                    result['P']['reason'], flags=re.IGNORECASE)
    checks = {'state_stability_not_overclaimed': all(result[k]['score'] is None and not result[k]['evidence_ids']
                                                    for k in ('AR','AL')),
              'readback': abs(read['h2_wt_percent'] - p[-1][-3]) < .03 and abs(read['dehydrogenation_kj_per_mol_h2'] - h) < .01
                          and read['in_40_70_window'] is True,
              'valid_changed_bond': bool(sites) and all(len(s) == 2 and all(type(i) is int and 0 <= i < n for i in s)
                                                        and p[s[0]][n+s[1]] != 0 for s in sites),
              'grounded_reaction_score': type(result['P']['score']) in (int, float) and 0 <= result['P']['score'] <= 100
                                        and set(result['P']['evidence_ids']) == set(references),
              'state_sites_valid': all(result[k]['sites'] == [] for k in ('AR', 'AL')),
              'cycle_evidence_scope': result['P'].get('cycle_evidence') == 'unknown' and not re.search(
                  r'\b(?:fully|completely|proven|confirmed)\s+reversib|\b(?:proves|confirms|establishes)\s+reversib',
                  reason, re.IGNORECASE)}
    save(out / 'results.json', {'result': result, 'checks': checks, 'passed': all(checks.values())})
    print(checks, flush=True)
    if not all(checks.values()):
        raise ValueError('Critic output did not pass validation')
    save(out / 'scoring_memory.json', {'matrices': enriched, 'assessment': result, 'checks': checks,
        'evidence': [{'id': r['id'], 'doi': r['doi'], 'anchor': r['anchor']} for r in memory],
        'model': load(out / 'critic_audit.json')['model'], 'prompt_sha256': sha256(instruction.encode()).hexdigest()})


def score_workflow(out, api, query, memory, sources):
    pair = matrix_pair(query)
    matching = [r for r in memory if matrix_pair(r['matrices']) == pair]
    def enthalpies(records):
        return {r['value'] for r in records if r['scope'] == 'reaction_enthalpy'
                and r['direction'] == 'rich_to_lean' and r['unit'] == 'kJ/mol H2'
                and type(r['value']) in (int, float) and math.isfinite(r['value'])}

    values = enthalpies(matching)
    if len(values) > 1:
        raise ValueError('Conflicting enthalpy observations require review before scoring')
    usable = bool(values)
    save(out / 'workflow.json', {'matrix_pair': pair, 'memory_hit': usable, 'search_invoked': not usable,
                                'matched_observations': len(matching)})
    if not usable:
        search(out, api, assisted=True, matrices=query)
        if not sources:
            print('Search completed; source excerpts await review before admission and scoring.', flush=True)
            return
        assess(out, api, sources, memory_only=True)
        matching = load(out / 'memory.json')
        values = enthalpies(matching)
    if len(values) != 1:
        raise ValueError('Expected one verified dehydrogenation enthalpy')
    h, = values
    enriched = encode(pair, h)
    save(out / 'enriched_matrices.json', enriched)
    score(out, api, enriched, matching)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--assist-search', action='store_true')
    parser.add_argument('--memory-only', action='store_true')
    parser.add_argument('--score-query', type=Path)
    parser.add_argument('--memory', type=Path)
    args = parser.parse_args()
    api = DeepSeek(credential(args.key_file), args.out)
    if args.score_query:
        args.out.mkdir(parents=True, exist_ok=False)
        query = load(args.score_query)
        save(args.out / 'query_matrices.json', query)
        score_workflow(args.out, api, query, load(args.memory) if args.memory else [],
                       load(args.sources) if args.sources else [])
    elif args.sources:
        assess(args.out, api, load(args.sources), memory_only=args.memory_only)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        search(args.out, api, assisted=args.assist_search)
