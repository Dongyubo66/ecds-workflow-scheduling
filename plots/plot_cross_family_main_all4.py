from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCHED_ORDER = ['HEFT', 'GREENHEFT', 'MOHEFT', 'ECDS']
SCHED_COLORS = {
    'HEFT': '#4C78A8',
    'GREENHEFT': '#F58518',
    'MOHEFT': '#54A24B',
    'ECDS': '#E45756',
}
SHORT_INSTANCE_LABELS = {
    'Montage-01d': 'M-01d',
    'Montage-05d': 'M-05d',
    'Epigenomics-50k': 'Epi-50k',
    'Epigenomics-100k': 'Epi-100k',
    'Seismology-100p': 'Sei-100p',
    'Seismology-200p': 'Sei-200p',
}
BROKEN_AXIS_INSTANCE = 'Montage-05d'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='dynamic_main_summary_all4.csv')
    p.add_argument('--out-dir', required=True, help='figure output directory')
    p.add_argument(
        '--col-only',
        action='store_true',
        help='only write the column-friendly broken-axis figures',
    )
    return p.parse_args()


def configure_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
    })


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_stem: str,
    add_title: bool = False,
) -> None:
    labels = df['instance_label'].drop_duplicates().tolist()
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(SCHED_ORDER))

    errkw = dict(
        ecolor='#2F2F2F',
        elinewidth=1.5,
        capthick=1.5,
    )

    for offset, sched in zip(offsets, SCHED_ORDER):
        sdf = df[df['scheduler'] == sched].copy()
        sdf = sdf.set_index('instance_label').reindex(labels).reset_index()

        means = sdf[f'{metric}_mean'].astype(float).to_numpy()
        stds = sdf[f'{metric}_std'].astype(float).fillna(0.0).to_numpy()

        ax.bar(
            x + offset,
            means,
            width=width,
            label=sched,
            yerr=stds,
            capsize=6,
            error_kw=errkw,
            color=SCHED_COLORS[sched],
            edgecolor='#2F2F2F',
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha='right')
    ax.set_ylabel(ylabel)

    if add_title:
        ax.set_title(out_stem.replace('_', ' '))

    ax.grid(axis='y', linestyle='--', alpha=0.25, zorder=0)

    ax.legend(
        ncol=2,
        frameon=True,
        loc='upper right',
        borderpad=0.4,
        handlelength=1.8,
    )

    ax.set_axisbelow(True)
    fig.tight_layout()

    pdf_path = Path(out_stem + '.pdf')
    png_path = Path(out_stem + '.png')

    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)


def _broken_axis_limits(df: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    mean_col = f'{metric}_mean'
    non_broken = df[df['instance_label'] != BROKEN_AXIS_INSTANCE][mean_col].astype(float)
    broken = df[df['instance_label'] == BROKEN_AXIS_INSTANCE][mean_col].astype(float)

    if non_broken.empty:
        raise ValueError(f'No non-{BROKEN_AXIS_INSTANCE} values found for {metric}')
    if broken.empty:
        raise ValueError(f'No {BROKEN_AXIS_INSTANCE} values found for {metric}')

    lower_upper = 1.15 * float(non_broken.max())
    upper_lower = 0.98 * float(broken.min())
    upper_upper = 1.03 * float(broken.max())

    if lower_upper >= upper_lower:
        raise ValueError(
            f'Broken-axis ranges overlap for {metric}: '
            f'lower upper={lower_upper:.4g}, upper lower={upper_lower:.4g}'
        )

    return lower_upper, upper_lower, upper_upper


def _add_break_marks(ax_upper, ax_lower) -> None:
    d = 0.012
    kwargs = dict(color='black', clip_on=False, linewidth=0.6)

    ax_upper.plot((-d, +d), (-d, +d), transform=ax_upper.transAxes, **kwargs)
    ax_upper.plot((1 - d, 1 + d), (-d, +d), transform=ax_upper.transAxes, **kwargs)
    ax_lower.plot((-d, +d), (1 - d, 1 + d), transform=ax_lower.transAxes, **kwargs)
    ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_lower.transAxes, **kwargs)


def plot_metric_col_broken_axis(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_stem: str,
) -> None:
    labels = df['instance_label'].drop_duplicates().tolist()
    short_labels = [SHORT_INSTANCE_LABELS.get(label, label) for label in labels]
    x = np.arange(len(labels))
    width = 0.16
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(SCHED_ORDER))

    lower_upper, upper_lower, upper_upper = _broken_axis_limits(df, metric)

    with plt.rc_context({
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        fig, (ax_upper, ax_lower) = plt.subplots(
            2,
            1,
            figsize=(3.35, 2.76),
            sharex=True,
            gridspec_kw={'height_ratios': [1.0, 2.05], 'hspace': 0.06},
        )

        errkw = dict(
            ecolor='#2F2F2F',
            elinewidth=0.55,
            capthick=0.55,
        )

        for ax in (ax_upper, ax_lower):
            for offset, sched in zip(offsets, SCHED_ORDER):
                sdf = df[df['scheduler'] == sched].copy()
                sdf = sdf.set_index('instance_label').reindex(labels).reset_index()

                means = sdf[f'{metric}_mean'].astype(float).to_numpy()
                stds = sdf[f'{metric}_std'].astype(float).fillna(0.0).to_numpy()

                ax.bar(
                    x + offset,
                    means,
                    width=width,
                    label=sched if ax is ax_upper else '_nolegend_',
                    yerr=stds,
                    capsize=1.6,
                    error_kw=errkw,
                    color=SCHED_COLORS[sched],
                    edgecolor='#2F2F2F',
                    linewidth=0.35,
                    zorder=3,
                )

            ax.set_xlim(-0.55, len(labels) - 0.45)
            ax.grid(axis='y', linestyle='-', linewidth=0.35, alpha=0.28, zorder=0)
            ax.grid(axis='x', visible=False)
            ax.tick_params(axis='both', width=0.6, length=2.0, labelsize=7)
            ax.set_axisbelow(True)

        ax_lower.set_ylim(0, lower_upper)
        ax_upper.set_ylim(upper_lower, upper_upper)

        ax_upper.spines['bottom'].set_visible(False)
        ax_lower.spines['top'].set_visible(False)
        ax_upper.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_lower.tick_params(axis='x', top=False)

        ax_lower.set_xticks(x)
        ax_lower.set_xticklabels(short_labels, rotation=25, ha='right', rotation_mode='anchor')
        fig.text(0.055, 0.52, ylabel, rotation='vertical', va='center', ha='left', fontsize=8)

        ax_upper.legend(
            ncol=2,
            frameon=True,
            loc='upper right',
            borderpad=0.25,
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.75,
        )

        _add_break_marks(ax_upper, ax_lower)

        fig.subplots_adjust(left=0.165, right=0.985, bottom=0.15, top=0.985)
        png_path = Path(out_stem + '.png')
        fig.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)


def main():
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_style()

    df = pd.read_csv(inp)
    needed = [
        'instance_label', 'scheduler',
        'brown_energy_mean', 'brown_energy_std',
        'total_carbon_mean', 'total_carbon_std',
        'green_ratio_mean', 'green_ratio_std',
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df['scheduler'] = df['scheduler'].astype(str).str.upper()
    df = df[df['scheduler'].isin(SCHED_ORDER)].copy()

    if not args.col_only:
        plot_metric(
            df=df,
            metric='brown_energy',
            ylabel='Brown energy (model units)',
            out_stem=str(out_dir / 'fig_dynamic_cross_family_brown_all4_journal'),
            add_title=False,
        )

        plot_metric(
            df=df,
            metric='total_carbon',
            ylabel='Total carbon (model units)',
            out_stem=str(out_dir / 'fig_dynamic_cross_family_carbon_all4_journal'),
            add_title=False,
        )

        plot_metric(
            df=df,
            metric='green_ratio',
            ylabel='Green ratio (fraction)',
            out_stem=str(out_dir / 'fig_dynamic_cross_family_green_ratio_all4_journal'),
            add_title=False,
        )

    plot_metric_col_broken_axis(
        df=df,
        metric='brown_energy',
        ylabel='Brown energy (model units)',
        out_stem=str(out_dir / 'fig_dynamic_brown_cross_family_col'),
    )

    plot_metric_col_broken_axis(
        df=df,
        metric='total_carbon',
        ylabel='Total carbon (model units)',
        out_stem=str(out_dir / 'fig_dynamic_carbon_cross_family_col'),
    )

    print(f'[OK] Saved figures to: {out_dir}')


if __name__ == '__main__':
    main()
