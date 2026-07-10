from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path.cwd()
DATA_DIR = ROOT / 'results' / 'final_submission_snapshot'
OUT_DIR = DATA_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ABL_IN = DATA_DIR / 'ablation_summary.csv'
TAG_ORDER = ['full', 'no_resched', 'no_brown', 'no_cluster', 'no_cluster_no_brown']
SCHED_TO_TAG = {
    'ECDS': 'full',
    'ECDS-NORESCHED': 'no_resched',
    'ECDS-NOBROWN': 'no_brown',
    'ECDS-NOCLUSTER': 'no_cluster',
    'ECDS-NOCLUSTER-NOBROWN': 'no_cluster_no_brown',
}


def short_instance_label(instance: str) -> str:
    s = str(instance).lower()
    if 'montage' in s and '01d' in s:
        return 'Montage-01d'
    if 'montage' in s and '05d' in s:
        return 'Montage-05d'
    if 'montage' in s and '10d' in s:
        return 'Montage-10d'
    return instance


def _load_malformed_ablation_csv(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        cleaned.append(line)

    lvl0 = next(csv.reader([cleaned[0]]))
    lvl1 = next(csv.reader([cleaned[1]]))
    lvl2 = next(csv.reader([cleaned[2]]))

    cols = []
    for i, (a, b) in enumerate(zip(lvl0, lvl1)):
        if i < 3:
            cols.append(lvl2[i].strip())
        else:
            a = a.strip()
            b = b.strip()
            if a and b:
                cols.append(f'{a}_{b}')
            elif a:
                cols.append(a)
            elif b:
                cols.append(b)
            else:
                cols.append(f'col_{i}')

    rows = []
    for raw in cleaned[3:]:
        vals = next(csv.reader([raw]))
        if len(vals) < len(cols):
            vals += [''] * (len(cols) - len(vals))
        elif len(vals) > len(cols):
            vals = vals[:len(cols)]
        rows.append(vals)

    return pd.DataFrame(rows, columns=cols)


def load_ablation_df(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if {'scenario', 'instance', 'scheduler'}.issubset(df.columns):
            return df
    except Exception:
        pass
    return _load_malformed_ablation_csv(path)


def plot_metric(df: pd.DataFrame, metric: str, output_name: str) -> None:
    df = df.copy()
    df['tag'] = df['scheduler'].astype(str).str.upper().map(SCHED_TO_TAG)
    df = df[df['tag'].isin(TAG_ORDER)].copy()

    if 'scenario' in df.columns:
        dyn = df[df['scenario'].astype(str).str.lower() == 'dynamic'].copy()
        if not dyn.empty:
            df = dyn

    df['instance_label'] = df['instance'].map(short_instance_label)
    rep = df[df['instance_label'].isin(['Montage-05d', 'Montage-10d', 'Montage-01d'])].copy()
    if not rep.empty:
        df = rep

    df[metric] = pd.to_numeric(df[metric], errors='coerce')

    labels = df['instance_label'].drop_duplicates().tolist()
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    offsets = np.linspace(-2 * width, 2 * width, num=len(TAG_ORDER))

    for offset, tag in zip(offsets, TAG_ORDER):
        t = df[df['tag'] == tag].copy()
        t = t.set_index('instance_label').reindex(labels).reset_index()
        vals = t[metric].astype(float).to_numpy()
        ax.bar(x + offset, vals, width=width, label=tag)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'Ablation comparison ({metric})')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / output_name, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    if not ABL_IN.exists():
        raise FileNotFoundError(f'Missing: {ABL_IN}')
    df = load_ablation_df(ABL_IN)

    for metric, out in [
        ('HV_mean', 'fig_ablation_hv.png'),
        ('IGD_mean', 'fig_ablation_igd.png'),
    ]:
        if metric in df.columns:
            plot_metric(df, metric, out)

    print(f'[OK] Saved ablation figures to: {OUT_DIR}')


if __name__ == '__main__':
    main()
