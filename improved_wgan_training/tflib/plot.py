import collections
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DPI = 800
COLOR_MAIN = '#1565C0'
COLOR_ALT = '#4FC3F7'

LABELS = {
    'time': ('Iter', 'Time, s'),
    'train disc cost': ('Iter', 'D loss'),
    'js1': ('Iter', 'JSD-1'),
    'js2': ('Iter', 'JSD-2'),
    'js3': ('Iter', 'JSD-3'),
    'js4': ('Iter', 'JSD-4'),
}

_OUTPUT_DIR = '.'
_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_iter = [0]


def init(output_dir):
    global _OUTPUT_DIR
    _OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)


def tick():
    _iter[0] += 1


def plot(name, value):
    _since_last_flush[name][_iter[0]] = value


def _style_axes(ax, name):
    xlabel, ylabel = LABELS.get(name, ('Iter', name))
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _save_metric_plot(name, x_vals, y_vals):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    color = COLOR_ALT if name == 'time' else COLOR_MAIN
    ax.plot(x_vals, y_vals, color=color, linewidth=1.2)
    _style_axes(ax, name)
    fig.tight_layout()
    filename = name.replace(' ', '_') + '.png'
    fig.savefig(os.path.join(_OUTPUT_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def flush():
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]
        _save_metric_plot(name, x_vals, y_vals)

    print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()

    with open(os.path.join(_OUTPUT_DIR, 'log.pkl'), 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)


def save_summary(title='Training'):
    if not _since_beginning:
        log_path = os.path.join(_OUTPUT_DIR, 'log.pkl')
        if not os.path.exists(log_path):
            return
        with open(log_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = dict(_since_beginning)

    metrics = [
        ('train disc cost', COLOR_MAIN),
        ('time', COLOR_ALT),
        ('js1', COLOR_MAIN),
        ('js2', COLOR_ALT),
        ('js3', COLOR_MAIN),
        ('js4', COLOR_ALT),
    ]
    present = [(name, color) for name, color in metrics if name in data and data[name]]
    if not present:
        return

    n = len(present)
    fig, axes = plt.subplots(n, 1, figsize=(6, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, color) in zip(axes, present):
        x_vals = np.sort(list(data[name].keys()))
        y_vals = [data[name][x] for x in x_vals]
        ax.plot(x_vals, y_vals, color=color, linewidth=1.2)
        _style_axes(ax, name)

    axes[-1].set_xlabel('Iter', fontsize=9)
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTPUT_DIR, 'training_summary.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
