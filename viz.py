# viz.py
import numpy as np
import matplotlib.pyplot as plt
import model


def fig_series_with_band(year, s: dict, title: str, y_label: str, target: float | None = None, y0: bool = True) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(year, s["mean"])
    ax.fill_between(year, s["lo"], s["hi"], alpha=0.2)
    if target is not None:
        ax.axhline(float(target), linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if y0:
        ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    return fig


def fig_class_split_over_time(year, sp_stats: dict, title: str, show_band: bool) -> plt.Figure:
    fig, ax = plt.subplots()
    for lab in model.STATE_LABELS:
        s = sp_stats[lab]
        ax.plot(year, s["mean"], label=lab)
        if show_band:
            ax.fill_between(year, s["lo"], s["hi"], alpha=0.15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of species total")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def fig_stacked_abundance(year, series_by_label: dict, title: str, y_label: str, show_band: bool = False) -> plt.Figure:
    fig, ax = plt.subplots()
    bottoms = np.zeros_like(year, dtype=float)
    total_lo = np.zeros_like(year, dtype=float)
    total_hi = np.zeros_like(year, dtype=float)

    for lab in model.STATE_LABELS:
        s = series_by_label[lab]
        vals = np.asarray(s["mean"], dtype=float)
        ax.bar(year, vals, bottom=bottoms, label=lab)
        bottoms = bottoms + vals
        total_lo = total_lo + np.asarray(s["lo"], dtype=float)
        total_hi = total_hi + np.asarray(s["hi"], dtype=float)

    if show_band:
        ax.fill_between(year, total_lo, total_hi, alpha=0.15)

    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def fig_stacked_species_totals(year, totals_stats: dict, title: str, y_label: str, species_display: dict[str, str] | None = None) -> plt.Figure:
    species_display = species_display or {}
    fig, ax = plt.subplots()
    bottoms = np.zeros_like(year, dtype=float)
    for sp in model.SPECIES:
        s = totals_stats[f"{sp}_total"]
        vals = np.asarray(s["mean"], dtype=float)
        ax.bar(year, vals, bottom=bottoms, label=species_display.get(sp, sp))
        bottoms = bottoms + vals
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    return fig
