"""Evidence gaps and scoped empirical disagreement, not score confidence."""
from collections import Counter,defaultdict
import hashlib,json,math
import numpy as np


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()[:20]


def family(pair):
    import networkx as nx
    values=[]
    for m in pair:
        g=nx.Graph();g.add_nodes_from(range(len(m)))
        g.add_edges_from((i,j) for i in range(len(m)) for j in range(i+1,len(m)) if m[i,j,7])
        ring=g.copy();ring.remove_edges_from(list(nx.bridges(g)))
        values.append(sorted((len(c),ring.subgraph(c).number_of_edges(),sorted(int(m[i,i,0]) for i in c),
            sorted(float(m[i,j,7]) for i,j in ring.subgraph(c).edges()))
            for c in nx.connected_components(ring) if ring.subgraph(c).number_of_edges()))
    return digest(values)


def local_key(pair,center,radius=2):
    import networkx as nx
    blocks=[]
    for m in pair:
        g=nx.Graph()
        for i in range(len(m)):g.add_node(i)
        for i in range(len(m)):
            for j in range(i+1,len(m)):
                if m[i,j,7]:g.add_edge(i,j)
        nodes=nx.single_source_shortest_path_length(g,center,cutoff=radius)
        labels={i:digest([m[i,i,:7].tolist(),i==center,len(list(g.neighbors(i)))]) for i in nodes}
        for _ in range(radius+1):
            labels={i:digest([labels[i],sorted((float(m[i,j,7]),labels[j]) for j in g.neighbors(i) if j in nodes)]) for i in nodes}
        blocks.append(labels[center])
    return digest(blocks)


def entropy(labels):
    counts=Counter(labels);n=sum(counts.values())
    return -sum((v/n)*math.log(v/n) for v in counts.values()) if n else None


class MemoryKnowledge:
    def __init__(self):
        self.family_sources=defaultdict(set);self.local_sources=defaultdict(set)
        self.evaluated_relations=set();self.judgments=defaultdict(dict)
        self.evidence_units=set();self.evaluated_molecules=set();self.contexts=set()

    def add(self,record,pair):
        f=family(pair)
        for c in record.get('evidence',{}).get('claims',[]):
            # A paper duplicated across streams is not a new independent source.
            unit=(c['source_id'],c['endpoint'],c.get('target_state'),c.get('conditions',''))
            self.evidence_units.add(unit);self.family_sources[(f,c['endpoint'])].add(unit)
            for site in c.get('site_parent_ids',[]):
                if 0<=site<pair.shape[1]:self.local_sources[(f,local_key(pair,site),c['endpoint'])].add(unit)
        evaluation=record.get('evaluation',{})
        assessment=evaluation.get('assessment')
        if not assessment:return
        context=evaluation.get('score_context','unversioned')
        uid=assessment['molecule_id'];self.evaluated_molecules.add(uid)
        for finding in assessment['findings']:
            scoped=context+'|conditions:'+' '.join(finding.get('conditions','unspecified').lower().split())
            self.contexts.add(scoped)
            sites=finding.get('site_parent_ids',[])
            for site in sites:
                if not 0<=site<pair.shape[1]:continue
                key=(f,local_key(pair,site),finding['endpoint'],scoped)
                self.evaluated_relations.add(key)
                # Deduplicate each molecule+finding+stream; do not count copies.
                item=(uid,assessment['stream'],finding['finding_id'])
                self.judgments[key][item]=finding['direction']

    def describe(self,pair,sites):
        f=family(pair);tokens=set();disagreements=[];details=[]
        endpoints={'stability':'storage_stability','reversibility':'cycle_reversibility'}
        for site in sites:
            local=local_key(pair,site)
            for endpoint,source_endpoint in endpoints.items():
                global_sources=self.family_sources.get((f,source_endpoint),set())
                local_sources=self.local_sources.get((f,local,source_endpoint),set())
                known_eval=any((f,local,endpoint,c) in self.evaluated_relations for c in self.contexts)
                if not global_sources:tokens.add(('whole_evidence',f,endpoint))
                if not local_sources:tokens.add(('local_evidence',f,endpoint,local))
                if not known_eval:tokens.add(('unassessed_relation',f,endpoint,local))
                # Different contexts never enter the same entropy calculation.
                for context in self.contexts:
                    values=list(self.judgments.get((f,local,endpoint,context),{}).values())
                    if values:
                        h=entropy(values);disagreements.append(dict(endpoint=endpoint,site=site,context=context,
                            empirical_judgment_entropy=h,judgments=len(values),
                            independent_experimental_evidence=False))
                        if h>0:tokens.add(('judgment_disagreement',f,endpoint,local,context))
                details.append(dict(site=site,endpoint=endpoint,whole_source_count=len(global_sources),
                    local_source_count=len(local_sources),has_evaluated_relation=known_eval))
        return dict(family=f,gap_tokens=sorted(tokens),coverage_gap_count=len(tokens),
            scoped_disagreement=disagreements,details=details)


def build_frontier(candidates,knowledge,known_ids,max_distance=.8):
    result=[]
    for c in candidates:
        if c['uid'] in known_ids or not c.get('valid',False) or c['anchor_distance']>max_distance:continue
        info=knowledge.describe(c['pair'],c['query_sites'])
        h=max((v['empirical_judgment_entropy'] for v in info['scoped_disagreement']),default=0.)
        if info['coverage_gap_count'] or h>0:
            result.append(dict(c,knowledge=info))
    return result


def allocate_frontier(frontier,budget,operator_posteriors=None,beta=1.):
    """Greedy unique-gap coverage + operator reward UCB, NOT calibrated entropy gain."""
    post=operator_posteriors or {};covered=set();selected=[];audit=[]
    for _ in range(min(budget,len(frontier))):
        options=[]
        for i,c in enumerate(frontier):
            if i in selected:continue
            tokens=set(tuple(t) for t in c['knowledge']['gap_tokens'])
            gain=len(tokens-covered)
            arm=post.get(c['operator'],dict(mean=0.,variance=0.))
            ucb=arm['mean']+beta*math.sqrt(max(arm['variance'],0))
            # Token coverage is a transparent prototype exploration utility.
            options.append((gain+ucb,gain,c['uid'],i,tokens,ucb))
        _,gain,_,i,tokens,ucb=max(options,key=lambda x:(x[0],x[1],x[2]))
        selected.append(i);covered|=tokens
        audit.append(dict(uid=frontier[i]['uid'],operator=frontier[i]['operator'],
            new_knowledge_gap_tokens=gain,operator_reward_ucb=ucb,
            reason='Unique unanswered structural relations; no score interval is read'))
    return [frontier[i] for i in selected],audit
