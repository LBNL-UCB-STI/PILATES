from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from link_inspector import (
    compute_link_neighborhood,
    detect_link_id_col,
    extract_linkstats_subset,
    extract_network_rows,
    plot_attribute_map,
    plot_link_hour_timeseries,
    plot_neighborhood_map,
    save_tables,
)


def _default_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    data_dir = here.parent / "network-calibration" / "data"
    return (
        data_dir / "5.linkstats.csv",
        data_dir / "seattle-area-cbg120-ferry-weakConn-network--car-only.geojson",
    )


def parse_args() -> argparse.Namespace:
    default_linkstats, default_network = _default_paths()
    parser = argparse.ArgumentParser(
        description="Skew-friendly diagnostics for link-level congestion proxies (density)."
    )
    parser.add_argument(
        "--linkstats",
        type=Path,
        default=default_linkstats,
        help="Path to *.linkstats.csv (expects link/hour stats).",
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=default_network,
        help="Path to the network GeoJSON (for mapping results).",
    )
    parser.add_argument(
        "--network-link-id-col",
        type=str,
        default=None,
        help="Column in the network file that matches linkstats 'link' ids (default: auto-detect).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scratch/network-calibration/output/calibration-eval"),
        help="Output directory for plots and CSV summaries.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="How many links to include in 'top links' visualizations.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.995,
        help="Quantile clip for color scaling (skew-friendly).",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=None,
        help="Optional absolute density threshold for flagging (same units as computed).",
    )
    parser.add_argument(
        "--density-threshold-quantile",
        type=float,
        default=0.999,
        help="If --density-threshold is not set, use this quantile of max_density to flag links.",
    )
    parser.add_argument(
        "--min-hours-over-threshold",
        type=int,
        default=2,
        help="Flag only links with at least this many hours above threshold.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip network map output (useful if geopandas backend is missing).",
    )
    parser.add_argument(
        "--inspect-link",
        type=int,
        default=None,
        help="If set, zoom in on a specific link id and write neighborhood tables/plots.",
    )
    parser.add_argument(
        "--inspect-hops",
        type=int,
        default=2,
        help="Node-adjacency hops for --inspect-link neighborhood extraction.",
    )
    parser.add_argument(
        "--inspect-top-n-timeseries",
        type=int,
        default=12,
        help="How many neighborhood links to include in the hour-by-hour timeseries plot.",
    )
    parser.add_argument(
        "--inspect-zoom-out",
        type=float,
        default=1.6,
        help="Zoom-out multiplier for neighborhood maps (larger = more context).",
    )
    parser.add_argument(
        "--inspect-basemap",
        action="store_true",
        help="Attempt to add a basemap for neighborhood maps (requires contextily + tile access).",
    )
    return parser.parse_args()


def load_linkstats(path: Path) -> pd.DataFrame:
    usecols = [
        "link",
        "hour",
        "freespeed",
        "capacity",
        "volume",
        "traveltime",
        "length",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["link"] = pd.to_numeric(df["link"], errors="raise").astype("int64")
    df["hour"] = pd.to_numeric(df["hour"], errors="raise").astype("int64")

    # vht: vehicle-hours on the link during the hour (traveltime assumed seconds, volume vehicles)
    df["vht"] = df["traveltime"] * df["volume"] / 3600.0
    # proxy density: vehicle-hours per meter (highly skewed, used here as a congestion diagnostic)
    df["density"] = df["vht"] / df["length"].replace(0, np.nan)

    # Helpful secondary diagnostics for "unphysical" patterns.
    df["speed"] = df["length"].replace(0, np.nan) / df["traveltime"].replace(0, np.nan)
    df["speed_ratio"] = df["speed"] / df["freespeed"].replace(0, np.nan)
    df["vc_ratio"] = df["volume"] / df["capacity"].replace(0, np.nan)
    df["freeflow_time"] = df["length"].replace(0, np.nan) / df["freespeed"].replace(
        0, np.nan
    )
    df["delay_time"] = (df["traveltime"] - df["freeflow_time"]).clip(lower=0)
    # Vehicle-seconds of delay over the hour (good for spillback / system impact).
    df["delay"] = df["delay_time"] * df["volume"]
    df["delay_per_meter"] = df["delay"] / df["length"].replace(0, np.nan)
    df["delay_time_per_meter"] = df["delay_time"] / df["length"].replace(0, np.nan)

    df = df.set_index(["link", "hour"]).sort_index()
    return df


def summarize_per_link(ls: pd.DataFrame) -> pd.DataFrame:
    grouped = ls.groupby(level="link")
    summary = grouped["density"].agg(
        max_density="max",
        p95_density=lambda s: s.quantile(0.95),
        mean_density="mean",
    )
    summary["peak_hour"] = (
        ls["density"].groupby(level="link").idxmax().map(lambda x: x[1])
    )
    summary["min_speed_ratio"] = grouped["speed_ratio"].min()
    summary["min_speed"] = grouped["speed"].min()
    summary["max_vc_ratio"] = grouped["vc_ratio"].max()
    summary["max_delay"] = grouped["delay"].max()
    summary["p95_delay"] = grouped["delay"].quantile(0.95)
    summary["max_delay_per_meter"] = grouped["delay_per_meter"].max()
    summary["p95_delay_per_meter"] = grouped["delay_per_meter"].quantile(0.95)
    return summary.sort_values("max_density", ascending=False)


def add_threshold_flags(
    per_link: pd.DataFrame,
    ls: pd.DataFrame,
    density_threshold: float | None,
    density_threshold_quantile: float,
    min_hours_over_threshold: int,
) -> tuple[pd.DataFrame, float]:
    if density_threshold is None:
        density_threshold = float(
            per_link["max_density"].quantile(density_threshold_quantile)
        )
    hours_over = (
        ls["density"]
        .reset_index()
        .assign(over=lambda d: d["density"] >= density_threshold)
        .groupby("link")["over"]
        .sum()
        .rename("hours_over_threshold")
    )
    out = per_link.join(hours_over, how="left")
    out["hours_over_threshold"] = out["hours_over_threshold"].fillna(0).astype(int)
    out["flag_suspect_density"] = out["hours_over_threshold"] >= int(
        min_hours_over_threshold
    )
    return out, density_threshold


def _require_matplotlib():
    try:
        if "MPLCONFIGDIR" not in os.environ:
            os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".mplconfig")
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401

        return
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotting requires matplotlib; install it in your env (e.g., conda install matplotlib)."
        ) from e


def plot_density_hist(ls: pd.DataFrame, out_dir: Path) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    densities = ls["density"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    positive = densities[densities > 0]
    if positive.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    log_bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 80)
    ax.hist(positive, bins=log_bins, color="#4C78A8", alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Link-hour density distribution (log-log)")
    ax.set_xlabel("density (vehicle-hours / meter)")
    ax.set_ylabel("count (link-hours)")
    fig.tight_layout()
    fig.savefig(out_dir / "density_hist_loglog.png", dpi=150)
    plt.close(fig)


def plot_top_links_heatmap(
    ls: pd.DataFrame,
    per_link: pd.DataFrame,
    out_dir: Path,
    top_n: int,
    clip_quantile: float,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    top_links = per_link.head(top_n).index
    mat = ls.loc[(top_links, slice(None)), "density"].unstack(level="hour")
    mat = mat.reindex(index=top_links)  # preserve sorted by severity

    values = mat.replace([np.inf, -np.inf], np.nan).to_numpy().astype(float)
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return
    vmin = max(float(np.nanmin(finite)), 1e-9)
    vmax = float(np.nanquantile(finite, clip_quantile))
    vmax = max(vmax, vmin)

    fig, ax = plt.subplots(figsize=(12, max(6, int(top_n * 0.18))))
    im = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_title(f"Top {top_n} links by max density (rows) across hours (cols)")
    ax.set_xlabel("hour")
    ax.set_ylabel("link id")
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(mat.columns.tolist())
    yticks = np.arange(values.shape[0])
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(i) for i in mat.index.tolist()])
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("density (vehicle-hours / meter)")
    fig.tight_layout()
    fig.savefig(out_dir / "top_links_density_heatmap.png", dpi=150)
    plt.close(fig)


def plot_network_map(
    net: gpd.GeoDataFrame,
    per_link: pd.DataFrame,
    link_id_col: str,
    out_dir: Path,
    clip_quantile: float,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    joined = net.merge(
        per_link[["max_density", "hours_over_threshold", "flag_suspect_density"]],
        left_on=link_id_col,
        right_index=True,
        how="left",
    )
    values = joined["max_density"].replace([np.inf, -np.inf], np.nan)
    positive = values[(values > 0) & values.notna()]
    if positive.empty:
        return

    vmin = max(float(positive.min()), 1e-9)
    vmax = float(positive.quantile(clip_quantile))
    vmax = max(vmax, vmin)

    fig, ax = plt.subplots(figsize=(12, 12))
    joined.plot(
        ax=ax,
        column="max_density",
        cmap="inferno",
        linewidth=0.6,
        legend=True,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        missing_kwds={"color": "#d3d3d3", "linewidth": 0.2, "label": "missing"},
    )
    ax.set_axis_off()
    ax.set_title("Max density per link (log color, clipped)")
    fig.tight_layout()
    fig.savefig(out_dir / "network_max_density_map.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ls = load_linkstats(args.linkstats)
    per_link = summarize_per_link(ls)
    per_link, density_threshold = add_threshold_flags(
        per_link=per_link,
        ls=ls,
        density_threshold=args.density_threshold,
        density_threshold_quantile=args.density_threshold_quantile,
        min_hours_over_threshold=args.min_hours_over_threshold,
    )

    per_link.to_csv(args.out / "per_link_density_summary.csv", index=True)
    flagged = per_link[per_link["flag_suspect_density"]].copy()
    flagged.to_csv(args.out / "flagged_links.csv", index=True)

    plot_density_hist(ls, args.out)
    plot_top_links_heatmap(
        ls, per_link, args.out, top_n=args.top_n, clip_quantile=args.clip_quantile
    )

    # Load network if we need mapping or inspection outputs.
    net = None
    link_id_col = None
    if (not args.no_map) or (args.inspect_link is not None):
        net = gpd.read_file(args.network)
        link_id_col = detect_link_id_col(net, args.network_link_id_col)

        if not args.no_map:
            plot_network_map(
                net, per_link, link_id_col, args.out, clip_quantile=args.clip_quantile
            )

        if args.inspect_link is not None:
            neighborhood = compute_link_neighborhood(
                net,
                int(args.inspect_link),
                hops=int(args.inspect_hops),
                link_id_col=link_id_col,
            )
            net_subset = extract_network_rows(
                net, neighborhood.link_ids, link_id_col=link_id_col
            )
            net_subset["_link_id"] = pd.to_numeric(
                net_subset[link_id_col], errors="coerce"
            ).astype("Int64")
            net_subset["is_target"] = net_subset["_link_id"] == int(args.inspect_link)
            if (
                "linkLength" in net_subset.columns
                and "numberOfLanes" in net_subset.columns
            ):
                net_subset["storage_veh"] = (
                    pd.to_numeric(net_subset["linkLength"], errors="coerce")
                    * pd.to_numeric(net_subset["numberOfLanes"], errors="coerce")
                    / 7.5
                )
            net_subset = net_subset.merge(
                per_link[
                    [
                        "max_density",
                        "p95_density",
                        "mean_density",
                        "peak_hour",
                        "hours_over_threshold",
                        "flag_suspect_density",
                        "min_speed_ratio",
                        "max_vc_ratio",
                        "max_delay",
                        "p95_delay",
                        "max_delay_per_meter",
                        "p95_delay_per_meter",
                    ]
                ],
                left_on="_link_id",
                right_index=True,
                how="left",
            ).drop(columns=["_link_id"])
            linkstats_subset = extract_linkstats_subset(ls, neighborhood.link_ids)

            tag = f"inspect_link_{int(args.inspect_link)}_h{int(args.inspect_hops)}"
            inspect_dir = args.out / "inspect" / tag
            save_tables(
                out_dir=inspect_dir,
                tag=tag,
                neighborhood=neighborhood,
                net_subset=net_subset,
                linkstats_subset=linkstats_subset,
                link_id_col=link_id_col,
            )

            if not args.no_map:
                plot_neighborhood_map(
                    net_subset,
                    out_path=inspect_dir / f"{tag}_map.png",
                    link_id_col=link_id_col,
                    target_link_id=int(args.inspect_link),
                    severity_by_link=per_link["max_density"],
                    clip_quantile=args.clip_quantile,
                    zoom_out_factor=float(args.inspect_zoom_out),
                    basemap=bool(args.inspect_basemap),
                )
                plot_attribute_map(
                    net_subset,
                    out_path=inspect_dir / f"{tag}_roadtype_lanes_map.png",
                    link_id_col=link_id_col,
                    target_link_id=int(args.inspect_link),
                    category_col="attributeOrigType",
                    lanes_col="numberOfLanes",
                    zoom_out_factor=float(args.inspect_zoom_out),
                    basemap=bool(args.inspect_basemap),
                )
                plot_link_hour_timeseries(
                    ls,
                    out_path=inspect_dir / f"{tag}_density_timeseries.png",
                    link_ids=neighborhood.link_ids,
                    value_col="density",
                    top_n=int(args.inspect_top_n_timeseries),
                    label_links=[int(args.inspect_link)],
                )

    print(f"Wrote outputs to: {args.out}")
    print(f"Density threshold used for flagging: {density_threshold:g}")
    print(
        f"Flagged links: {len(flagged)} (min hours over threshold: {args.min_hours_over_threshold})"
    )
    if len(flagged) > 0:
        print("Top flagged links (by max_density):")
        cols = [
            "max_density",
            "hours_over_threshold",
            "peak_hour",
            "min_speed_ratio",
            "max_vc_ratio",
            "max_delay",
        ]
        print(
            flagged[cols]
            .head(min(20, len(flagged)))
            .sort_values("max_delay")
            .to_string()
        )
    if args.inspect_link is not None:
        tag = f"inspect_link_{int(args.inspect_link)}_h{int(args.inspect_hops)}"
        print(f"Inspect outputs: {args.out / 'inspect' / tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
