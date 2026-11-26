"""Complete the four original visualization tasks. Keep plotting separate from fitting.

Every function in this module only DRAWS. The data, fitted models and probe
grids are prepared by ``main.py``; here we never train or modify a model.
This separation is intentional: each figure stays reproducible from the same
inputs, and the plotting code is easy to explain in the report.
"""

import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Shared visual style
# ---------------------------------------------------------------------------
# scikit-learn fixes the Iris class order as 0=setosa, 1=versicolor,
# 2=virginica. Using the SAME colours everywhere lets the reader follow one
# species across all six figures without re-reading a legend.
CLASS_COLORS = ("#E46F61", "#5BA88B", "#6C8EBF")      # strong marker colours
REGION_COLORS = ("#F8DAD4", "#D5EBDF", "#DCE6F2")      # pale decision regions
REGION_CMAP = ListedColormap(REGION_COLORS)
POINT_CMAP = ListedColormap(CLASS_COLORS)
# One sequential map per species: white means P=0, the species colour means
# P=1. Three maps keep the species identity visible in the probability panels.
PROB_CMAPS = tuple(
    LinearSegmentedColormap.from_list(f"p{k}", ["#FFFFFF", color])
    for k, color in enumerate(CLASS_COLORS)
)

# Fixed scales. Contour levels must be explicit: if we let matplotlib choose
# them from the data it would stretch 0..1 over whatever range happens to be
# present, and the panels would no longer be comparable.
PROB_LEVELS = np.linspace(0.0, 1.0, 11)
REGION_LEVELS = np.array([-0.5, 0.5, 1.5, 2.5])  # class IDs 0,1,2 sit mid-bin


def _style_2d(ax, x_label, y_label):
    """Apply axis labels and a light grid shared by the 2D figures."""
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, color="#DDDDDD", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)  # grid behind the data, never on top


# Task #1, slides 43–46 — 2D decision regions
def plot_regions(ax, xx, yy, region, X, y):
    """Draw predicted class regions and overlay the measured flowers.

    xx, yy : meshgrid of probe coordinates in centimetres (x3 = petal length,
        x4 = petal width), each of shape (120, 160).
    region : the winning class ID at every probe, already predicted by main.py.
    X, y   : all 150 measured flowers and their observed class IDs.

    The pale background is the model's ANSWER at each point; the dark dots are
    the GROUND TRUTH. Where a dot disagrees with the patch under it, the model
    made a mistake -- exactly what the report should discuss.
    """
    # contourf paints one flat colour per predicted class. The explicit levels
    # guarantee that label 0/1/2 map onto REGION_COLORS[0/1/2].
    ax.contourf(xx, yy, region, levels=REGION_LEVELS,
                cmap=REGION_CMAP, alpha=0.95)

    # White edges keep every flower legible even on a similarly coloured patch.
    # vmin/vmax pin the colour scale to 0..2 so the mapping never drifts.
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=POINT_CMAP, vmin=0, vmax=2,
               s=38, edgecolors="white", linewidths=0.9,
               label="Measured flowers", zorder=3)

    _style_2d(ax, r"Petal length $x_3$ (cm)", r"Petal width $x_4$ (cm)")
    ax.set_xlim(0.6, 7.3)
    ax.set_ylim(-0.1, 2.8)

    swatches = [Patch(facecolor=c, edgecolor="white") for c in CLASS_COLORS]
    ax.legend(swatches, ["setosa", "versicolor", "virginica"],
              loc="upper left", fontsize=9, framealpha=0.9, title="Species")


# Task #1, slides 47–49 — probability maps for each classifier
def plot_probabilities(axes, xx, yy, probabilities, class_names):
    """Draw one probability panel per species, all on a shared 0–1 scale.

    probabilities has shape (n_y, n_x, 3); layer k holds P(class k) at every
    probe. All three panels use IDENTICAL contour levels and colorbar limits,
    otherwise confidence cannot be compared across species or classifiers.
    """
    fig = axes[0].figure

    for k, ax in enumerate(axes):
        layer = probabilities[..., k]  # class_names[k] labels layer k directly

        im = ax.contourf(xx, yy, layer, levels=PROB_LEVELS,
                         cmap=PROB_CMAPS[k], vmin=0.0, vmax=1.0)

        # A colourbar per panel turns "darker = higher" into actual numbers.
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(f"P({class_names[k]})", fontsize=12)
        _style_2d(ax, r"$x_3$ petal length (cm)", r"$x_4$ petal width (cm)")
        ax.set_xlim(0.6, 7.3)
        ax.set_ylim(-0.1, 2.8)

    for ax in axes[1:]:
        ax.set_yticklabels([])  # only the left panel needs y tick labels


# Task #2, slides 50–52 — binary decision plane
def plot_boundary_3d(ax, model, X, y):
    """Draw the observed flowers and the plane where P(virginica) = 0.5.

    The binary model was fitted on the 100 non-setosa flowers in RAW
    centimetres, so coef_/intercept_ are used directly -- no scaler here.

    For logistic regression the log-odds are linear:
        s = w2*x2 + w3*x3 + w4*x4 + b ,  P(virginica) = sigmoid(s)
    P = 0.5 happens exactly when s = 0, i.e. on the plane
        x4 = -(w2*x2 + w3*x3 + b) / w4 .
    Solving for x4 turns an abstract boundary into a surface we can inspect.
    """
    # 1. the measured flowers (y holds 1=versicolor, 2=virginica)
    for cls, name in ((1, "versicolor"), (2, "virginica")):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                   c=CLASS_COLORS[cls], label=name, s=34,
                   edgecolors="white", linewidths=0.6, alpha=0.9)

    # 2. the 0.5 decision plane
    w2, w3, w4 = model.coef_[0]
    b = model.intercept_[0]
    x2_grid = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 40)
    x3_grid = np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 40)
    g2, g3 = np.meshgrid(x2_grid, x3_grid)
    g4 = -(w2 * g2 + w3 * g3 + b) / w4  # w4 is nonzero for the fitted model
    ax.plot_surface(g2, g3, g4, color="#9AA0A6", alpha=0.45,
                    linewidth=0, antialiased=True)

    ax.set_xlabel(r"Sepal width $x_2$ (cm)")
    ax.set_ylabel(r"Petal length $x_3$ (cm)")
    ax.set_zlabel(r"Petal width $x_4$ (cm)")
    ax.set_title("Binary logistic model: the 0.5 decision plane")
    ax.view_init(elev=23, azim=-58)
    ax.legend(loc="upper left", fontsize=9)


# Task #3, slide 53 — probabilities in 3D
def plot_probability_3d(ax, model, X, y):
    """Show how P(virginica) varies through the three-feature space.

    A regular 3D grid of probes is queried in one batch, and each probe is
    coloured by its probability on a FIXED 0–1 scale: warm = confident
    virginica, cool = confident versicolor.

    model.classes_ is [1, 2] -> column 0 is versicolor, column 1 virginica.
    Looking the column up by value avoids the classic off-by-one bug.
    """
    # 1. probe grid covering the observed feature ranges
    n = 14  # 14^3 = 2744 probes, dense enough to show a smooth gradient
    x2 = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, n)
    x3 = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, n)
    x4 = np.linspace(X[:, 2].min() - 0.1, X[:, 2].max() + 0.1, n)
    probe = np.stack(np.meshgrid(x2, x3, x4, indexing="ij"), axis=-1).reshape(-1, 3)

    # 2. one batched query; pick the virginica column by its class value
    p = model.predict_proba(probe)
    virginica_col = list(model.classes_).index(2)
    q = p[:, virginica_col]

    # 3. coloured probes
    sc = ax.scatter(probe[:, 0], probe[:, 1], probe[:, 2],
                    c=q, cmap="plasma", vmin=0.0, vmax=1.0,
                    s=16, alpha=0.65, edgecolors="none")

    # 4. measured flowers on top, in dark grey with white edges
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c="#222222", s=30,
               edgecolors="white", linewidths=0.7,
               label="Measured flowers", depthshade=False)

    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label(r"$q = P(\mathrm{virginica}\mid x_2,x_3,x_4)$")

    ax.set_xlabel(r"Sepal width $x_2$ (cm)")
    ax.set_ylabel(r"Petal length $x_3$ (cm)")
    ax.set_zlabel(r"Petal width $x_4$ (cm)")
    ax.set_title("Probability of virginica in the 3-feature space")
    ax.view_init(elev=23, azim=-58)
    ax.legend(loc="upper left", fontsize=9)


# Task #4, slides 54–57 — optional hook for a saved-state extension
def draw_frame(ax, state, grid, shape, X, y):
    """Draw one decision-region frame from a saved (W, b) pair.

    ``state`` is one snapshot from learning_states.npz: W has shape (2, 3),
    b has shape (3). The probe grid is already in the standardized coordinate
    system used during training, so logits are simply ``grid @ W + b`` and the
    winning class is the argmax per row. X holds standardized petal features.

    Keeping this separate makes it trivial to render any epoch on demand,
    which is exactly what an animation (or a grid of stills) needs.
    """
    W, b = state
    logits = grid @ W + b                       # (n_probes, 3) raw scores
    labels = logits.argmax(axis=1).reshape(shape)

    ax.contourf(np.linspace(0.6, 7.3, shape[1]),
                np.linspace(-0.1, 2.8, shape[0]),
                labels, levels=REGION_LEVELS, cmap=REGION_CMAP, alpha=0.95)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=POINT_CMAP, vmin=0, vmax=2,
               s=26, edgecolors="white", linewidths=0.7)
    _style_2d(ax, r"Petal length $x_3$ (std)", r"Petal width $x_4$ (std)")
