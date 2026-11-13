"""Plot cumulative C−A structure counts and visual-only local score differences."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager, ticker
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'LOHC_From_Zero_Parallel_v10'
sys.path.insert(0, str(ROOT))
from structural_projection import StructuralProjection
from compare_pooled_percentiles import canonical, load


def first_evaluation(run, arm, cap):
    seen = {}
    for path in sorted((run / arm).glob('graph_evaluations_*.json')):
        for row in load(path):
            if row['index'] > cap or not row['new_valid_evaluation']:
                continue
            pair = canonical(row['smi']), canonical(row['rich_proxy_smi'])
            if None not in pair:
                seen.setdefault(pair, row['index'])
    return seen


def prepare(run):
    report = load(run / 'pooled_comparison.json')
    cap = report['gnn_common_cap']
    pool = {tuple(row['pair']): row for row in load(run / 'pooled_molecules.json')}
    seen = {arm: first_evaluation(run, arm, cap) for arm in ('native', 'llm')}
    for arm, pairs in seen.items():
        if pairs.keys() != {p for p, row in pool.items() if arm in row['arms']}:
            raise ValueError(f'{arm} candidate records disagree with the pooled results')
    histories = {arm: {r['generation']: r['gnn_pairs'] for r in load(run / arm / 'progress_history.json')}
                 for arm in seen}
    checkpoints = [min(h[g] for h in histories.values()) for g in (4, 8)] + [cap]
    projection = StructuralProjection(run)
    return count_frames(projection, pool, seen, checkpoints)


def count_frames(projection, pool, seen, checkpoints):
    projection.fit()
    smiles = sorted({p[0] for p in pool})
    vectors = np.stack([projection.features(smi) for smi in smiles])
    points = dict(zip(smiles, projection.model.transform(vectors)))
    coordinates = np.stack(list(points.values()))
    step = .035
    edges = [np.arange(np.floor(v.min()/step)*step-4*step,
                       np.ceil(v.max()/step)*step+5*step, step) for v in coordinates.T]
    centers = [(axis[:-1] + axis[1:]) / 2 for axis in edges]
    frames = []
    for limit in checkpoints:
        groups = {}
        for arm, records in seen.items():
            selected = [p for p, index in records.items() if index <= limit]
            xy = np.stack([points[p[0]] for p in selected])
            score = np.asarray([pool[p]['score'] if pool[p]['score'] is not None else np.nan for p in selected])
            rated = np.isfinite(score)
            counts = np.histogram2d(*xy.T, bins=edges)[0]
            n_scored = np.histogram2d(*xy[rated].T, bins=edges)[0]
            sum_score = np.histogram2d(*xy[rated].T, bins=edges, weights=score[rated])[0]
            if (counts.sum(), n_scored.sum()) != (len(selected), rated.sum()):
                raise ValueError('Candidates lost during binning')
            smooth = [gaussian_filter(v, sigma=1.2, mode='constant', truncate=3) for v in (counts, n_scored, sum_score)]
            if not np.isclose(smooth[0].sum(), counts.sum()):
                raise ValueError('Smoothed counts are not mass-preserving')
            mean = np.divide(smooth[2], smooth[1], out=np.full_like(counts, np.nan),
                             where=(smooth[1] >= 1) & (smooth[1] >= .25*smooth[0]))
            groups[arm] = (counts, n_scored, smooth[0], mean)
        a, c = groups['native'], groups['llm']
        height = c[2] - a[2]
        if not np.isclose(height.sum(), c[0].sum() - a[0].sum()):
            raise ValueError('Signed difference failed conservation check')
        frames.append({'cap': limit, 'a': a, 'c': c, 'height': height,
                       'supported': (c[2]+a[2]) >= .25, 'color': c[3]-a[3]})
    if len(frames) == 3 and not np.allclose(frames[0]['height'], 0):
        raise ValueError('The common warmup differed unexpectedly')
    return centers, frames


def draw(centers, frames, out):
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
    x, y = np.meshgrid(*centers, indexing='ij')
    zlimit = max(1, float(max(np.max(np.abs(f['height'])) for f in frames))) * 1.08
    color_limit = max(1, float(max(np.nanmax(np.abs(f['color'])) for f in frames)))
    cmap = colors.LinearSegmentedColormap.from_list('score_delta', ['#286286', '#f8f7f1', '#a94828'])
    norm = colors.TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit)
    fig = plt.figure(figsize=(14, 6.1), facecolor='white')
    grid = fig.add_gridspec(1, 3, left=.045, right=.94, bottom=.22, top=.78, wspace=.05)
    fig.text(.06, .935, 'C−A 分子生成差异', fontsize=24, weight='bold', color='#16303a')
    fig.text(.06, .875, '共同结构坐标  ·  按相同 GNN 评价预算回放', fontsize=12, color='#486169')
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(grid[0, i], projection='3d', proj_type='ortho')
        rgba = cmap(norm(np.nan_to_num(frame['color'])))
        rgba[~np.isfinite(frame['color'])] = colors.to_rgba('#abb4ba')
        ax.plot_surface(x, y, np.where(frame['supported'], frame['height'], np.nan), facecolors=rgba,
                        rstride=1, cstride=1, linewidth=0, shade=False, antialiased=False)
        ax.view_init(elev=29, azim=-62)
        ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()), zlim=(-zlimit, zlimit),
               xlabel='结构 PC1', ylabel='结构 PC2')
        if i == 0:
            ax.set_zlabel('C−A 数量差', labelpad=4)
        ax.set_box_aspect((1.10, 1.2, .78))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(ticker.MaxNLocator(4))
            axis.set_pane_color((.95, .97, .97, 1))
        ax.tick_params(labelsize=8, pad=0)
        a, c = (int(frame[k][0].sum()) for k in ('a', 'c'))
        name = ('共同预热结束', '搜索中期', '最终共同预算')[i]
        ax.set_title(f'{name}  ·  {frame["cap"]:,} 次 GNN 评价\nA {a:,} 对   /   C {c:,} 对', fontsize=11, pad=3)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.24,.129,.41,.022]),
                       orientation='horizontal', ticks=[-color_limit, 0, color_limit], format='%.1f')
    bar.set_label('C−A 局部近似均分差（分，仅作可视化）', labelpad=5)
    fig.legend(handles=[Patch(facecolor='#abb4ba', label='共同评分支撑不足')], loc='center left',
               bbox_to_anchor=(.68,.14), frameon=False, fontsize=9)
    fig.text(.06,.040,'高度为相同结构网格内累计唯一有效分子对数之差，经保总量平滑。正值表示 C 生成更多。',
             fontsize=9,color='#486169')
    fig.savefig(out / 'ca_count_surface.png', dpi=220)
    fig.savefig(out / 'ca_count_surface.pdf')
    plt.close(fig)


def main(run, out):
    centers, frames = prepare(run)
    out.mkdir(parents=True, exist_ok=True)
    draw(centers, frames, out)
    with (out / 'ca_count_surface_grid.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['gnn_cap','pc1','pc2','a_count','c_count','a_scored','c_scored',
                         'raw_count_delta','surface_count_delta','visual_mean_delta','surface_visible'])
        for f in frames:
            a, c = f['a'], f['c']
            for i, j in np.ndindex(f['height'].shape):
                writer.writerow([f['cap'],centers[0][i],centers[1][j],int(a[0][i,j]),int(c[0][i,j]),
                                 int(a[1][i,j]),int(c[1][i,j]),int(c[0][i,j]-a[0][i,j]),f['height'][i,j],
                                 f['color'][i,j] if np.isfinite(f['color'][i,j]) else '',bool(f['supported'][i,j])])
    print(json.dumps([{'cap': f['cap'], 'A': int(f['a'][0].sum()), 'C': int(f['c'][0].sum()),
                       'peak': float(f['height'].max()), 'trough': float(f['height'].min()),
                       'colored_visible_cells': int((np.isfinite(f['color']) & f['supported']).sum())} for f in frames]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    main(args.run, args.out)
