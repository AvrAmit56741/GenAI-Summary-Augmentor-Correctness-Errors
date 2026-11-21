"""Figure for the erasure experiment (exp_layers.py erase / erase orth) -> figures/layers_erase.png"""
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
cols = ['negation', 'number', 'agreement_suffix', 'other']
short = {'agreement_suffix': 'agreement', 'all_types': 'shared (all types)'}
for a, (f, title, rows) in zip(ax, [
        ('layers_erase_summary.csv', '(a) Full type direction erased', ['all_types', 'negation', 'number', 'agreement_suffix', 'random']),
        ('layers_erase_summary_orth.csv', '(b) Type-specific part only', ['negation', 'number', 'agreement_suffix', 'random'])]):
    t = pd.read_csv(f); p = t.pivot(index='dir', columns='rule', values='flagged')
    D = (p.loc[rows, cols] - p.loc['none', cols]).values
    im = a.imshow(D, cmap='RdBu', vmin=-1, vmax=1)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            a.text(j, i, f'{D[i, j]:+.2f}', ha='center', va='center', fontsize=9)
    n = t.groupby('rule').n.first()
    a.set_xticks(range(len(cols)), [f'{short.get(c, c)}\n(n={n[c]})' for c in cols], fontsize=9)
    a.set_yticks(range(len(rows)), [short.get(r, r) for r in rows], fontsize=9)
    a.set(title=title, xlabel='test sentences (error type)')
ax[0].set_ylabel('erased direction')
fig.suptitle('Change in share of corrupted sentences flagged after erasing a direction at all 28 layers (vs. no erasure)')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.savefig('figures/layers_erase.png', dpi=160, bbox_inches='tight')
