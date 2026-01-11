import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, List

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


def figtext_along_3d_line(fig, ax, p0, p1, text, offset_px=(0, 0), **text_kwargs):
    """
    Places fig-level text aligned with the projected direction of the 3D segment p0->p1.
    offset_px shifts the label in screen pixels (dx, dy).
    """
    # Project 3D to 2D display coordinates
    x0, y0, _ = proj3d.proj_transform(*p0, ax.get_proj())
    x1, y1, _ = proj3d.proj_transform(*p1, ax.get_proj())

    x0d, y0d = ax.transData.transform((x0, y0))
    x1d, y1d = ax.transData.transform((x1, y1))

    # Angle of the segment in screen space
    angle = np.degrees(np.arctan2((y1d - y0d), (x1d - x0d)))

    # Midpoint in display coords + optional pixel offset
    xm = (x0d + x1d) / 2 + offset_px[0]
    ym = (y0d + y1d) / 2 + offset_px[1]

    # Convert display coords -> figure fraction coords
    xf, yf = fig.transFigure.inverted().transform((xm, ym))

    fig.text(xf, yf, text, rotation=angle, rotation_mode='anchor', **text_kwargs)


def get_cube_faces(center: Sequence[float], size: float) -> List[List[List[float]]]:
    r = [center[i] - size / 2 for i in range(3)] + [center[i] + size / 2 for i in range(3)]
    return [
        [[r[0], r[1], r[2]], [r[3], r[1], r[2]], [r[3], r[4], r[2]], [r[0], r[4], r[2]]],
        [[r[0], r[1], r[5]], [r[3], r[1], r[5]], [r[3], r[4], r[5]], [r[0], r[4], r[5]]],
        [[r[0], r[1], r[2]], [r[3], r[1], r[2]], [r[3], r[1], r[5]], [r[0], r[1], r[5]]],
        [[r[0], r[4], r[2]], [r[3], r[4], r[2]], [r[3], r[4], r[5]], [r[0], r[4], r[5]]],
        [[r[0], r[1], r[2]], [r[0], r[4], r[2]], [r[0], r[4], r[5]], [r[0], r[1], r[5]]],
        [[r[3], r[1], r[2]], [r[3], r[4], r[2]], [r[3], r[4], r[5]], [r[3], r[1], r[5]]],
    ]


def build_custom_cmap(colors: Sequence[str], name: str = "custom_cmap", n: int = 256):
    return LinearSegmentedColormap.from_list(name, list(colors), N=n)

@dataclass(frozen=True)
class CubeConfig:
    size: float
    fc: str
    ec: str
    alpha: float


@dataclass(frozen=True)
class PlotConfig:
    fig_size: Tuple[int, int] = (12, 10)
    facecolor: str = "white"

    # Scene setup
    limit: float = 0.8
    cube_center: Tuple[float, float, float] = (0.0, -0.15, -0.1)

    # Dots
    n_dots: int = 500
    dot_seed: int = 42
    dot_range: float = 0.65
    dot_size: int = 6
    dot_color: str = "#2c6cb0"
    dot_alpha: float = 0.95

    # View
    elev: float = 20
    azim: float = -62

    cb_vmin: float = 0.15  # was 0.19
    cb_vmax: float = 0.65  # was 0.60
    cb_ticks: Tuple[float, ...] = (0.19, 0.3, 0.4, 0.5, 0.6)
    cb_axes_rect: Tuple[float, float, float, float] = (0.88, 0.15, 0.02, 0.6)


def create_figure_and_axes(cfg: PlotConfig):
    fig = plt.figure(figsize=cfg.fig_size, facecolor=cfg.facecolor)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def add_nested_cubes(ax, cfg: PlotConfig, cubes: Sequence[CubeConfig]):
    for i, c in enumerate(cubes):
        faces = get_cube_faces(cfg.cube_center, c.size)

        edge_color = c.ec
        edge_width = 0.5
        if i == 0:  # inner cube emphasis
            edge_color = "black"
            edge_width = 1.8

        poly = Poly3DCollection(
            faces,
            facecolors=c.fc,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=c.alpha,
        )
        ax.add_collection3d(poly)


def add_center_line(ax):
    ax.plot3D(
        [-0.28, 0.1],
        [0, 0],
        [-0.30, -0.30],
        color="black",
        linestyle="--",
        linewidth=3,
        zorder=30,
    )


def add_sparse_dots(ax, cfg: PlotConfig):
    rng = np.random.default_rng(cfg.dot_seed)
    dots = rng.uniform(-cfg.dot_range, cfg.dot_range, (cfg.n_dots, 3))
    ax.scatter(
        dots[:, 0],
        dots[:, 1],
        dots[:, 2],
        s=cfg.dot_size,
        color=cfg.dot_color,
        alpha=cfg.dot_alpha,
        zorder=20,
    )


def style_axes_grid(ax, cfg: PlotConfig):
    limit = cfg.limit
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")
    ax.grid(True, linestyle="-", color="gray", alpha=0.3, linewidth=1)

    ticks = np.linspace(-limit, limit, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def add_labels_and_arrows(fig, ax, cfg: PlotConfig):
    limit = cfg.limit

    # --- Labels ---
    # 1D in front (fig-level text)
    figtext_along_3d_line(
        fig,
        ax,
        (-limit, -0.30, -limit + 0.3),
        (limit, -0.30, -limit + 0.3),
        "1D",
        offset_px=(0, -12),
        fontsize=18,
        ha="center",
        va="center",
        color="black",
    )

    ax.text(-0.4, -0.9, -limit + 0.1, "2D", fontsize=18, ha="center", color="black")

    ax.text(limit - 0.05, -limit, -limit + 0.05, "3D", fontsize=18, ha="center", color="black")
    ax.text(0.7, 0.7, 0.7, "3D", fontsize=18, color="black")

    # --- Outer x-axis arrow ---
    y_arrow = -limit - 0.08
    z_arrow = -limit - 0.08
    ax.quiver(
        limit,
        y_arrow,
        z_arrow,
        -2 * limit,
        0,
        0,
        color="black",
        linewidth=2.5,
        arrow_length_ratio=0.03,
        pivot="tail",
        zorder=50,
    )

    # --- Length label (fig-level text along x) ---
    ax.annotate(
        "",
        xy=(0.22, 0.23),
        xytext=(0.58, 0.14),
        arrowprops=dict(arrowstyle="<->", color="black"),
    )

    figtext_along_3d_line(
        fig,
        ax,
        (-limit, -limit - 0.15, -limit - 0.15),
        (limit, -limit - 0.15, -limit - 0.15),
        "Length = 1",
        offset_px=(0, -10),
        fontsize=20,
        ha="center",
        va="center",
        color="black",
    )

    # --- Connected arrows: y then z (in front of Volume label) ---
    x_pos = limit + 0.10
    eps = 0.02

    ax.quiver(
        x_pos,
        limit - eps,
        -limit + eps,
        0,
        -(2 * limit - 2 * eps),
        0,
        color="black",
        linewidth=2.5,
        arrow_length_ratio=0.06,
        pivot="tail",
        zorder=50,
    )

    ax.quiver(
        x_pos,
        limit - eps,
        -limit + eps,
        0,
        0,
        2 * limit - 2 * eps,
        color="black",
        linewidth=2.5,
        arrow_length_ratio=0.06,
        pivot="tail",
        zorder=50,
    )

    # --- Volume label (fig-level text along y) ---
    ax.annotate(
        "",
        xy=(0.6, 0.16),
        xytext=(0.82, 0.35),
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    figtext_along_3d_line(
        fig,
        ax,
        (limit - 0.15, -limit, -limit - 0.15),
        (limit - 0.15, limit, -limit - 0.15),
        "Volume = 2",
        offset_px=(0, -10),
        fontsize=20,
        ha="center",
        va="center",
        color="black",
    )


def add_titles():
    plt.figtext(0.5, 0.96, "Curse of Dimensionality", fontsize=26, fontweight="bold", ha="center")
    plt.figtext(
        0.5,
        0.92,
        "Exponential Volume Growth + Data Sparsity + Distance Concentration",
        fontsize=16,
        ha="center",
    )


def add_colorbar(fig, cm, cfg: PlotConfig):
    # NOTE: edit cfg.cb_vmin/cfg.cb_vmax to move tick labels inward
    norm = Normalize(vmin=cfg.cb_vmin, vmax=cfg.cb_vmax)
    sm = ScalarMappable(norm=norm, cmap=cm)

    cb_ax = fig.add_axes(list(cfg.cb_axes_rect))
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_ticks(list(cfg.cb_ticks))
    cb.ax.tick_params(labelsize=14)
    cb.ax.set_title("Mean Distance\nfrom Center", fontsize=16, pad=20)
    return cb


def finalize(fig, ax, cfg: PlotConfig, out_path: str = "result.png", dpi: int = 150):
    ax.view_init(elev=cfg.elev, azim=cfg.azim)
    plt.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.1)
    fig.savefig(out_path, dpi=dpi)

def main(out_path: str):
    cfg = PlotConfig(
        cb_vmin=0.17,
        cb_vmax=0.63,
    )

    cubes = [
        CubeConfig(size=0.4, fc="#bfa600", ec="#7a6a00", alpha=1.0),
        CubeConfig(size=0.8, fc="#90ee90", ec="#70b040", alpha=0.3),
        CubeConfig(size=1.2, fc="#ade0f0", ec="#1e90ff", alpha=0.2),
    ]

    cmap = build_custom_cmap(["#1a2a8a", "#0160a0", "#00a08a", "#90d050", "#fefb3d"])

    fig, ax = create_figure_and_axes(cfg)
    add_nested_cubes(ax, cfg, cubes)
    add_center_line(ax)
    add_sparse_dots(ax, cfg)
    style_axes_grid(ax, cfg)
    add_labels_and_arrows(fig, ax, cfg)
    add_titles()
    add_colorbar(fig, cmap, cfg)
    finalize(fig, ax, cfg, out_path=out_path, dpi=150)


if __name__ == "__main__":
    out_dir = "curse_of_dimensionality_data_sparsity.png"
    main(out_dir)
