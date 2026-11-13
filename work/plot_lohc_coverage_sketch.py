"""Draw a comparison of occupied PCA cells and available structure scores."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'LOHC_From_Zero_Parallel_v10'
sys.path.insert(0, str(ROOT))
from structural_projection import StructuralProjection


CELL = 0.16
COUNT_COLORS = LinearSegmentedColormap.from_list('coverage', ['#edf2f5', '#a6bbc9', '#416079'])
SCORE_COLORS = mpl.colormaps['viridis']
SCORE_NORM = Normalize(0, 100)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def cells(projection, smiles):
    xy = np.array([projection.position(smi) for smi in smiles])
    return [tuple(v) for v in np.floor(xy / CELL).astype(int)]


def rectangle(cell, z):
    x, y = CELL * np.array(cell)
    return [(x, y, z), (x + CELL, y, z), (x + CELL, y + CELL, z), (x, y + CELL, z)]


def plot_3d(ax, counts, scores, norm, xlim, ylim):
    ground = Poly3DCollection([rectangle(cell, 0) for cell in counts],
                              facecolors=[COUNT_COLORS(norm(n)) for n in counts.values()],
                              edgecolor='#f8fafb', linewidth=.5)
    ax.add_collection3d(ground)
    for cell, values in scores.items():
        value = np.mean(values)
        color = SCORE_COLORS(SCORE_NORM(value))
        ax.add_collection3d(Poly3DCollection([rectangle(cell, value)], facecolor=color,
                                             edgecolor='#19354a', linewidth=.7))
        x, y = CELL * (np.array(cell) + .5)
        ax.plot([x, x], [y, y], [0, value], color='#587386', alpha=.43, linewidth=.7)
    ax.set(xlim=xlim, ylim=ylim, zlim=(0, 100), xlabel='结构 PC1', ylabel='结构 PC2', zlabel='结构潜力分')
    ax.set_xticks([-2, -1.5, -1, -.5, 0])
    ax.set_yticks([-.5, 0, .5, 1])
    ax.set_zticks([0, 25, 50, 75, 100])
    ax.set_box_aspect((1.15, 1, .70))
    ax.set_proj_type('ortho')
    ax.view_init(elev=27, azim=-56)
    ax.tick_params(labelsize=8, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('#dbe3e9')


def plot_plan(ax, counts, scores, norm, xlim, ylim):
    for cell, count in counts.items():
        x, y = CELL * np.array(cell)
        ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=COUNT_COLORS(norm(count)),
                               edgecolor='white', linewidth=.6))
        if cell in scores:
            score = np.mean(scores[cell])
            ax.add_patch(Rectangle((x + .014, y + .014), CELL - .028, CELL - .028,
                                   facecolor=SCORE_COLORS(SCORE_NORM(score)), edgecolor='#123745', linewidth=.7))
            ax.text(x + CELL / 2, y + CELL / 2, len(scores[cell]), color='white',
                    ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set(xlim=xlim, ylim=ylim, xlabel='结构 PC1', ylabel='结构 PC2', aspect='equal')
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(labelsize=8, length=2.5)


def main():
    trial = ROOT / 'structure_score_trial_01'
    run = ROOT / 'operator_ac_run_01'
    results = read(trial / 'results.json')
    protocol = read(trial / 'protocol.json')
    lookup = {row['id']: row for row in results['candidates']}
    projection = StructuralProjection(run)
    projection.fit()
    groups = {}
    for arm in ('native', 'llm'):
        generation = protocol['cutoffs'][arm]['generation']
        population = read(run / arm / f'parents_{generation:04d}.json')
        molecules = sorted({row['pair'][0] for row in population})
        counts = Counter(cells(projection, molecules))
        scored = [lookup[i] for i in results['membership'][arm] if lookup[i]['score'] is not None]
        scores = defaultdict(list)
        for cell, row in zip(cells(projection, [row['pair'][0] for row in scored]), scored):
            scores[cell].append(row['score'])
        assert set(scores) <= counts.keys() and sum(counts.values()) == len(molecules)
        groups[arm] = counts, scores, generation, len(molecules), len(scored)

    occupied = np.array(list(set().union(*(g[0] for g in groups.values()))))
    lower = occupied.min(axis=0) * CELL - .045
    upper = (occupied.max(axis=0) + 1) * CELL + .045
    xlim, ylim = list(zip(lower, upper))
    max_count = max(max(group[0].values()) for group in groups.values())
    norm = LogNorm(1, max_count)

    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'axes.labelcolor': '#304354', 'text.color': '#1d3040', 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(13.2, 9.3), facecolor='white')
    grid = fig.add_gridspec(2, 2, left=.08, right=.92, top=.84, bottom=.174, hspace=.20, wspace=.08,
                           height_ratios=(1.5, 1))
    for col, (arm, name) in enumerate((('native', 'A  原生 GA'), ('llm', 'C  LLM 耦合 UCB'))):
        counts, scores, gen, total, scored = groups[arm]
        ax = fig.add_subplot(grid[0, col], projection='3d')
        plot_3d(ax, counts, scores, norm, xlim, ylim)
        x = (.295, .705)[col]
        fig.text(x, .888, name, fontsize=15, fontweight='bold', ha='center')
        fig.text(x, .856, f'G{gen} · {total} 个合格父代 · 覆盖 {len(counts)} 格 · {scored} 个评分落入 {len(scores)} 格',
                 fontsize=10, color='#506575', ha='center')
        plan = fig.add_subplot(grid[1, col])
        plot_plan(plan, counts, scores, norm, xlim, ylim)
        plan.set_title('覆盖面的俯视图', fontsize=10, pad=6)
        print(arm, gen, total, len(counts), scored, len(scores), max(counts.values()))

    fig.text(.08, .959, '结构覆盖与评分', fontsize=25, fontweight='bold')
    fig.text(.08, .924, '底面显示候选覆盖范围和数量，高处面片显示已评分格的平均结构潜力分。',
             fontsize=12, color='#506575')
    for pos, cmap, scale, ticks, name in (
        ([.18, .080, .25, .016], COUNT_COLORS, norm, [1, 3, 10, 30, 100], '每格唯一候选数（对数色阶）'),
        ([.57, .080, .25, .016], SCORE_COLORS, SCORE_NORM, [0, 25, 50, 75, 100], '格内平均结构潜力分'),
    ):
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=scale, cmap=cmap), cax=fig.add_axes(pos),
                           orientation='horizontal', ticks=ticks)
        bar.ax.set_xticklabels([str(v) for v in ticks])
        bar.ax.tick_params(labelsize=8, length=2)
        bar.outline.set_visible(False)
        bar.ax.set_title(name, fontsize=9, pad=4)
    fig.text(.50, .027, '共用 PCA 和 0.16 × 0.16 网格。俯视图数字表示每格已评分分子数，空白表示该快照无候选。',
             ha='center', fontsize=9, color='#506575')
    out = trial / 'coverage_score_sketch.png'
    fig.savefig(out, dpi=175, facecolor='white')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
