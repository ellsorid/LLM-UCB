"""Hierarchical UCB over GA population policies, with separate score axes."""
from dataclasses import dataclass, field
from math import log, sqrt


@dataclass(frozen=True)
class Policy:
    name: str
    axis: str
    mix: tuple[int, int, int]   # crossover, guided mutation, immigration
    same_platform: bool
    preserve_scaffold: bool
    coverage_retention: float


POLICIES = (
    Policy("whole_crossover", "whole", (6,2,2), True,True,.30),
    Policy("whole_frontier", "whole", (4,2,4), False,False,.40),
    Policy("local_mutation", "local", (2,7,1), True,True,.25),
    Policy("local_balanced", "local", (4,5,1), True,True,.30),
)


def score_delta(parent, child):
    """Separate changes in observed score components from changes in blend weight."""
    pG,pL,pw = parent
    cG,cL,cw = child
    w = (pw+cw)/2
    return {
        "whole":(1-w)*(cG-pG),
        "local":w*(cL-pL),
        "evidence_weight_change":(cw-pw)*((pL+cL-pG-cG)/2),
    }


@dataclass
class History:
    count: int = 0
    value_sum: float = 0.

    def update(self, value):
        self.count += 1
        self.value_sum += value

    def mean(self, prior=0., strength=2.):
        return (self.value_sum+strength*prior)/(self.count+strength)


@dataclass
class PopulationUCB:
    coarse: dict = field(default_factory=lambda:{a:History() for a in ("whole","local")})
    fine: dict = field(default_factory=lambda:{p.name:History() for p in POLICIES})
    observed: set = field(default_factory=set)

    def observe(self, cohort_id, components, novel_cell_fraction=0., novel_unit_fraction=0.,
                policy=None, legal_yield=None):
        """One policy-tagged offspring cohort is one update, regardless of its size."""
        if cohort_id in self.observed or not components:
            raise ValueError("Cohort must be new and have validated feedback")
        if policy is not None and policy not in self.fine:
            raise ValueError("Unknown population policy")
        if legal_yield is not None and not 0<=legal_yield<=1:
            raise ValueError("Invalid legal yield")
        for value in (novel_cell_fraction,novel_unit_fraction):
            if not 0<=value<=1:
                raise ValueError("Coverage fraction outside 0–1")
        rewards = {
            "whole":sum(row["whole"] for row in components)/len(components)/100+.10*novel_cell_fraction,
            "local":sum(row["local"] for row in components)/len(components)/100+.10*novel_unit_fraction,
        }
        for ax, reward in rewards.items():
            self.coarse[ax].update(reward)
        if policy is not None:
            axis = next(p.axis for p in POLICIES if p.name==policy)
            self.fine[policy].update(rewards[axis] + (.05*legal_yield if legal_yield is not None else 0.))
        self.observed.add(cohort_id)
        return rewards

    def choose(self, gaps, beta=.1):
        """Cold policies borrow the coarse mean and retain exploration capacity."""
        if set(gaps)!={"whole","local"} or any(not 0<=v<=1 for v in gaps.values()):
            raise ValueError("Provide an evidence gap for both axes")
        step = len(self.observed)
        def bonus(count, axis):
            return beta*gaps[axis]*sqrt(log(step+2)/(count+1))
        coarse_ucb = {axis:h.mean()+bonus(h.count,axis) for axis,h in self.coarse.items()}
        axis = max(coarse_ucb,key=lambda a:(coarse_ucb[a],a=="whole"))
        fine_ucb = {p.name:self.fine[p.name].mean(self.coarse[axis].mean())+
                    bonus(self.fine[p.name].count,axis) for p in POLICIES if p.axis==axis}
        policy = max((p for p in POLICIES if p.axis==axis),key=lambda p:fine_ucb[p.name])
        return policy, {"coarse_ucb":coarse_ucb,"fine_ucb":fine_ucb}


def quotas(policy, budget):
    if budget<=0:
        raise ValueError("Population proposal budget must be positive")
    total = sum(policy.mix)
    raw = [budget*v/total for v in policy.mix]
    counts = [int(v) for v in raw]
    order = sorted(range(3),key=lambda i:(-(raw[i]-counts[i]),i))
    for i in order[:budget-sum(counts)]:
        counts[i] += 1
    return dict(zip(("crossover","mutation","immigration"),counts))
