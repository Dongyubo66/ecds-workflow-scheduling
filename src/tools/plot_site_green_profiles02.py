import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import yaml
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_profile(green_profile: List[dict]) -> List[Tuple[float, float]]:
    """
    Return sorted list of (start_time, green_fraction) with start_time as float.
    """
    prof = []
    for seg in (green_profile or []):
        if seg is None:
            continue
        s = float(seg.get("start", 0.0))
        g = float(seg.get("green", 0.0))
        g = max(0.0, min(1.0, g))
        prof.append((s, g))
    prof.sort(key=lambda x: x[0])
    return prof


def _infer_horizon(site_profiles: Dict[str, List[Tuple[float, float]]], default_extend: float = 80.0) -> float:
    mx = 0.0
    for _site, prof in site_profiles.items():
        if prof:
            mx = max(mx, max(t for t, _ in prof))
    return float(mx + default_extend)


def _step_xy(profile: List[Tuple[float, float]], horizon: float) -> Tuple[List[float], List[float]]:
    """
    Build x,y for ax.step(where='post'): x includes last horizon point.
    """
    if not profile:
        return [0.0, horizon], [0.0, 0.0]

    # Ensure starts begin at 0 for nicer visualization (optional)
    prof = profile[:]
    if prof[0][0] > 0.0:
        prof = [(0.0, prof[0][1])] + prof

    xs = [t for t, _ in prof]
    ys = [g for _, g in prof]

    # append horizon
    if horizon <= xs[-1]:
        horizon = xs[-1] + 1.0
    xs = xs + [horizon]
    ys = ys + [ys[-1]]
    return xs, ys


def _ci_from_g(g: float, ci_green: float, ci_brown: float) -> float:
    # Effective carbon intensity (kgCO2 per energy unit) under green fraction g
    return float(g) * float(ci_green) + (1.0 - float(g)) * float(ci_brown)


def _style_params(style: str) -> Dict[str, Any]:
    """
    paper: 默认适合论文（可彩色）
    patent: 黑白+线型区分+更“附图草稿”风格（代理人最终可能仍会重绘）
    """
    style = (style or "paper").strip().lower()
    if style == "patent":
        return dict(
            use_grid=False,
            linewidth=2.2,
            markers=False,
            legend_outside=True,
            monochrome=True,
            title_in_fig=False,
            dpi=600,
        )
    return dict(
        use_grid=True,
        linewidth=2.0,
        markers=True,
        legend_outside=False,
        monochrome=False,
        title_in_fig=True,
        dpi=300,
    )


def _line_style_cycle(monochrome: bool) -> List[dict]:
    # 专利：用线型区分；论文：默认也给线型，颜色由 matplotlib 自动分配
    base = [
        dict(linestyle="-"),
        dict(linestyle="--"),
        dict(linestyle=":"),
        dict(linestyle="-."),
    ]
    if monochrome:
        # 强制黑色，靠线型区分
        for b in base:
            b["color"] = "black"
    return base


def plot_all_sites_g(
    sites: Dict[str, dict],
    site_profiles: Dict[str, List[Tuple[float, float]]],
    outdir: Path,
    horizon: float,
    style: str = "paper",
    fmt: str = "png",
) -> Path:
    sp = _style_params(style)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.6))

    styles = _line_style_cycle(monochrome=sp["monochrome"])
    site_names = sorted(site_profiles.keys())

    for i, sname in enumerate(site_names):
        prof = site_profiles[sname]
        xs, ys = _step_xy(prof, horizon=horizon)
        st = styles[i % len(styles)].copy()
        if sp["markers"]:
            ax.step(xs, ys, where="post", linewidth=sp["linewidth"], marker="o", markersize=4, label=f"{sname} g(t)", **st)
        else:
            ax.step(xs, ys, where="post", linewidth=sp["linewidth"], label=f"{sname} g(t)", **st)

    ax.set_xlabel("time")
    ax.set_ylabel("green fraction g(t)")
    ax.set_xlim(left=0.0, right=float(horizon))
    ax.set_ylim(0.0, 1.0)

    if sp["use_grid"]:
        ax.grid(True, alpha=0.25)

    if sp["title_in_fig"]:
        ax.set_title("All sites: green fraction profiles g(t)")

    if sp["legend_outside"]:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        ax.legend(frameon=True)

    fpath = outdir / f"sites__green_fraction__all.{fmt}"
    fig.savefig(fpath, dpi=sp["dpi"], bbox_inches="tight")
    plt.close(fig)
    return fpath


def plot_all_sites_ci(
    sites: Dict[str, dict],
    site_profiles: Dict[str, List[Tuple[float, float]]],
    outdir: Path,
    horizon: float,
    style: str = "paper",
    fmt: str = "png",
) -> Path:
    sp = _style_params(style)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.6))

    styles = _line_style_cycle(monochrome=sp["monochrome"])
    site_names = sorted(site_profiles.keys())

    for i, sname in enumerate(site_names):
        prof = site_profiles[sname]
        ci_green = float(sites[sname].get("ci_green", 0.05))
        ci_brown = float(sites[sname].get("ci_brown", 0.55))

        xs, ys_g = _step_xy(prof, horizon=horizon)
        ys_ci = [_ci_from_g(g, ci_green, ci_brown) for g in ys_g]

        st = styles[i % len(styles)].copy()
        if sp["markers"]:
            ax.step(xs, ys_ci, where="post", linewidth=sp["linewidth"], marker="x", markersize=5,
                    label=f"{sname} CI(t)", **st)
        else:
            ax.step(xs, ys_ci, where="post", linewidth=sp["linewidth"], label=f"{sname} CI(t)", **st)

    ax.set_xlabel("time")
    ax.set_ylabel("effective carbon intensity CI(t)\n(kgCO2 per energy unit)")
    ax.set_xlim(left=0.0, right=float(horizon))

    if sp["use_grid"]:
        ax.grid(True, alpha=0.25)

    if sp["title_in_fig"]:
        ax.set_title("All sites: effective carbon intensity profiles CI(t)")

    if sp["legend_outside"]:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        ax.legend(frameon=True)

    fpath = outdir / f"sites__carbon_intensity__all.{fmt}"
    fig.savefig(fpath, dpi=sp["dpi"], bbox_inches="tight")
    plt.close(fig)
    return fpath


def plot_single_site_stacked(
    site_name: str,
    site_cfg: dict,
    profile: List[Tuple[float, float]],
    outdir: Path,
    horizon: float,
    style: str = "paper",
    fmt: str = "png",
) -> Path:
    """
    上下子图：上 g(t)，下 CI(t)，共享 x 轴（比 twinx 更适合专利/审稿阅读）
    """
    sp = _style_params(style)
    outdir.mkdir(parents=True, exist_ok=True)

    ci_green = float(site_cfg.get("ci_green", 0.05))
    ci_brown = float(site_cfg.get("ci_brown", 0.55))

    xs, ys_g = _step_xy(profile, horizon=horizon)
    ys_ci = [_ci_from_g(g, ci_green, ci_brown) for g in ys_g]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)

    styles = _line_style_cycle(monochrome=sp["monochrome"])
    st_g = styles[0].copy()
    st_ci = styles[1].copy()

    # g(t)
    if sp["markers"]:
        ax1.step(xs, ys_g, where="post", linewidth=sp["linewidth"], marker="o", markersize=4, label="g(t)", **st_g)
    else:
        ax1.step(xs, ys_g, where="post", linewidth=sp["linewidth"], label="g(t)", **st_g)
    ax1.set_ylabel("g(t)")
    ax1.set_ylim(0.0, 1.0)

    # CI(t)
    if sp["markers"]:
        ax2.step(xs, ys_ci, where="post", linewidth=sp["linewidth"], marker="x", markersize=5, label="CI(t)", **st_ci)
    else:
        ax2.step(xs, ys_ci, where="post", linewidth=sp["linewidth"], label="CI(t)", **st_ci)
    ax2.set_ylabel("CI(t)\n(kgCO2/energy)")
    ax2.set_xlabel("time")

    ax2.set_xlim(left=0.0, right=float(horizon))

    if sp["use_grid"]:
        ax1.grid(True, alpha=0.25)
        ax2.grid(True, alpha=0.25)

    if sp["title_in_fig"]:
        fig.suptitle(f"Site={site_name} | ci_green={ci_green}, ci_brown={ci_brown}")

    # legends (avoid covering curves)
    ax1.legend(loc="upper right", frameon=True)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fpath = outdir / f"site__{site_name}__g_and_ci.{fmt}"
    fig.savefig(fpath, dpi=sp["dpi"], bbox_inches="tight")
    plt.close(fig)
    return fpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/static.yaml", help="Path to YAML config (default: configs/static.yaml)")
    ap.add_argument("--outdir", default="results/plotsV2/sites", help="Output directory")
    ap.add_argument("--horizon", type=float, default=None, help="Time horizon for plotsV2 (default: auto)")
    ap.add_argument("--style", choices=["paper", "patent"], default="paper", help="Figure style preset")
    ap.add_argument("--fmt", choices=["png", "pdf"], default="png", help="Output format")
    ap.add_argument("--per_site", action="store_true", help="Also save per-site stacked figures")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    sites = ((cfg.get("resources") or {}).get("sites") or {})
    if not sites:
        raise ValueError("No resources.sites found in config. Please add resources: {sites: ...}")

    site_profiles: Dict[str, List[Tuple[float, float]]] = {}
    for sname, scfg in sites.items():
        prof = _normalize_profile((scfg or {}).get("green_profile") or [])
        site_profiles[str(sname)] = prof

    horizon = float(args.horizon) if args.horizon is not None else _infer_horizon(site_profiles)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    saved = []
    saved.append(plot_all_sites_g(sites, site_profiles, outdir, horizon=horizon, style=args.style, fmt=args.fmt))
    saved.append(plot_all_sites_ci(sites, site_profiles, outdir, horizon=horizon, style=args.style, fmt=args.fmt))

    if args.per_site:
        for sname in sorted(site_profiles.keys()):
            saved.append(
                plot_single_site_stacked(
                    sname,
                    sites[sname],
                    site_profiles[sname],
                    outdir,
                    horizon=horizon,
                    style=args.style,
                    fmt=args.fmt,
                )
            )

    print("-" * 80)
    print("Loaded config:", Path(args.config).resolve())
    print("Saved figures to:", outdir.resolve())
    print("Horizon:", horizon)
    print("Count:", len(saved))
    if saved:
        print("Example:", saved[0])


if __name__ == "__main__":
    main()
