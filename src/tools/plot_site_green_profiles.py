import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_sites(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    resources = cfg.get("resources", {}) or {}
    sites = resources.get("sites", {}) or {}
    if not isinstance(sites, dict) or not sites:
        raise ValueError("configs/static.yaml 中未找到 resources.sites 或为空。")
    return sites


def _step_xy_from_profile(profile: List[Dict[str, Any]], tmax: float) -> Tuple[List[float], List[float]]:
    """
    profile: [{start: 0, green: 0.6}, {start: 50, green: 0.9}, ...]
    返回用于 ax.step(where='post') 的 x, y
    """
    if not profile:
        return [0.0, float(tmax)], [0.0, 0.0]

    prof = sorted(profile, key=lambda d: float(d.get("start", 0.0)))
    starts = [float(d.get("start", 0.0)) for d in prof]
    greens = [float(d.get("green", 0.0)) for d in prof]

    # x 要包含最后一个 tmax（否则最后一段画不出来）
    x = starts + [float(tmax)]
    y = greens + [greens[-1]]
    return x, y


def _ci_from_g(g: float, ci_green: float, ci_brown: float) -> float:
    # CI(t) = g(t)*ci_green + (1-g(t))*ci_brown
    return float(g) * float(ci_green) + (1.0 - float(g)) * float(ci_brown)


def plot_all_sites_g(sites: Dict[str, Dict[str, Any]], outdir: Path, tmax: float, fmt_list: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    for site_name, sconf in sites.items():
        prof = sconf.get("green_profile", []) or []
        x, y = _step_xy_from_profile(prof, tmax)
        ax.step(x, y, where="post", linewidth=2.2, label=f"{site_name} g(t)")

    ax.set_title("All Sites | green fraction profiles", fontsize=14)
    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel("green fraction g(t)", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # legend 放图外更像论文图
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()

    _ensure_dir(outdir)
    for fmt in fmt_list:
        fpath = outdir / f"all_sites__green_fraction_profiles.{fmt}"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_site_g_and_ci(
    site_name: str,
    sconf: Dict[str, Any],
    outdir: Path,
    tmax: float,
    fmt_list: List[str],
) -> None:
    ci_green = float(sconf.get("ci_green", 0.05))
    ci_brown = float(sconf.get("ci_brown", 0.55))
    prof = sconf.get("green_profile", []) or []

    x, g = _step_xy_from_profile(prof, tmax)

    # 同步生成 CI 的 step y
    ci = [_ci_from_g(v, ci_green, ci_brown) for v in g]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    ax1.step(x, g, where="post", linewidth=2.4, label="green_fraction g(t)")
    ax1.set_xlabel("time", fontsize=12)
    ax1.set_ylabel("green fraction g(t)", fontsize=12)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(x, ci, where="post", linewidth=2.0, linestyle="--", label="effective carbon intensity CI(t)")
    ax2.set_ylabel("CI(t) (kgCO2 per energy unit)", fontsize=12)

    # 合并图例（避免左右各一个看起来乱）
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    ax1.set_title(f"Site={site_name} | ci_green={ci_green}, ci_brown={ci_brown}", fontsize=14)
    fig.tight_layout()

    _ensure_dir(outdir)
    for fmt in fmt_list:
        fpath = outdir / f"site__{site_name}__g_and_ci.{fmt}"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/static.yaml", help="YAML config path (default: configs/static.yaml)")
    ap.add_argument("--outdir", default="results/plotsV2/sites", help="Output dir (default: results/plotsV2/sites)")
    ap.add_argument("--tmax", type=float, default=200.0, help="Max time shown on x-axis (default: 200)")
    ap.add_argument("--formats", default="png", help="Comma-separated formats: png,pdf,svg (default: png)")
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path.resolve()}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    sites = _parse_sites(cfg)

    outdir = Path(args.outdir)
    fmt_list = [s.strip().lower() for s in str(args.formats).split(",") if s.strip()]
    tmax = float(args.tmax)

    # 统一一点论文风格的默认参数（不指定颜色）
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
    })

    plot_all_sites_g(sites, outdir, tmax, fmt_list)

    for site_name, sconf in sites.items():
        plot_site_g_and_ci(site_name, sconf, outdir, tmax, fmt_list)

    print("-" * 80)
    print("Loaded cfg:", cfg_path.resolve())
    print("Saved figures to:", outdir.resolve())
    print("Formats:", fmt_list)
    print("Sites:", ", ".join(sorted(sites.keys())))


if __name__ == "__main__":
    main()
