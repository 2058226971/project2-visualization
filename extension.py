"""Task #4 extension: replay the saved classifier states (no retraining).

The lecture offers two extension examples. This one chooses the "learning
replay" path: instead of fitting yet another model, we load the 301 snapshots
of weights and biases that were saved during training and redraw the SAME
probe grid at several epochs. The question answered is:

    How does the classifier's decision region move as training progresses?

Two artifacts are produced in ``outputs/``:
  * learning_replay.png -- a 2x2 still for the report (epochs 0, 15, 300 plus
    the train/validation accuracy curves);
  * learning_replay.gif  -- an animation sampling the whole training run.

Every state is used as-is. Nothing is retrained, so the replay is exact.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tasks import draw_frame, CLASS_COLORS

OUT = Path("outputs")
EPOCHS_FOR_STILLS = (0, 15, 300)  # early / middle / final, matches the lecture
GIF_STEP = 10                     # keep every 10th epoch for a light animation


def load_states():
    """Load the saved (W, b) snapshots and the accuracy history.

    weights[e] has shape (2, 3): two feature rows, three class columns.
    biases[e]  has shape (3,): one bias per class.
    The states consume STANDARDIZED petal features [length, width].
    """
    state = np.load(Path("data") / "learning_states.npz")
    weights, biases = state["weights"], state["biases"]
    with open(Path("data") / "learning_history.csv", newline="") as f:
        history = list(csv.DictReader(f))
    epoch = np.array([float(r["epoch"]) for r in history])
    train = np.array([float(r["train_accuracy"]) for r in history])
    val = np.array([float(r["val_accuracy"]) for r in history])
    return weights, biases, epoch, train, val


def build_probe_space():
    """Return (scaler, grid_standardized, shape, X_standardized, y).

    The probe grid spans the same raw petal coordinates as Task #1. It is
    pushed through the SAME scaler fitted on the supplied training split, so
    every state sees identical inputs -- only (W, b) changes from frame to
    frame. That is what makes the comparison of epochs meaningful.
    """
    iris = load_iris()
    X, y = iris.data[:, 2:4], iris.target
    train, _ = train_test_split(np.arange(len(y)), test_size=.3,
                                stratify=y, random_state=42)
    scaler = StandardScaler().fit(X[train])

    xx, yy = np.meshgrid(np.linspace(.6, 7.3, 160), np.linspace(-.1, 2.8, 120))
    grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    return scaler, grid, xx.shape, scaler.transform(X), y


def frame_state(weights, biases, e):
    """Bundle one epoch's parameters into the (W, b) tuple draw_frame expects."""
    return weights[e], biases[e]


def draw_accuracy(ax, epoch, train, val):
    """Plot both accuracy curves and annotate the three still epochs."""
    ax.plot(epoch, train, color=CLASS_COLORS[1], lw=1.8, label="Training")
    ax.plot(epoch, val, color=CLASS_COLORS[2], lw=1.8, label="Validation")
    for e in EPOCHS_FOR_STILLS:
        ax.axvline(e, color="#BBBBBB", lw=0.8, ls="--", zorder=0)
        ax.plot(e, val[e], "o", color=CLASS_COLORS[2], ms=6, zorder=3)
        ax.annotate(f"ep {e}\n{val[e]:.1%}", (e, val[e]),
                    textcoords="offset points", xytext=(8, -2),
                    fontsize=9, color="#333333")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.15, 1.02)
    ax.set_title("Accuracy across saved states")
    ax.grid(alpha=0.35)
    ax.legend(loc="lower right", fontsize=9)


def make_still(weights, biases, grid, shape, X, y, epoch, train, val):
    """Build the 2x2 figure used in the report."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, e in zip(axes.ravel()[:3], EPOCHS_FOR_STILLS):
        draw_frame(ax, frame_state(weights, biases, e), grid, shape, X, y)
        ax.set_title(f"Epoch {e}  (val acc {val[e]:.1%})", fontsize=11)
    draw_accuracy(axes.ravel()[3], epoch, train, val)
    fig.suptitle("Saved-state replay: the decision region moves as training proceeds",
                 fontsize=13)
    path = OUT / "learning_replay.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_gif(weights, biases, grid, shape, X, y):
    """Sample the training run into a short GIF animation.

    FuncAnimation redraws each frame through draw_frame, so the animation and
    the still figure share exactly the same plotting code. No retraining.
    """
    fig, ax = plt.subplots(figsize=(5.6, 4.2), layout="constrained")
    epochs = list(range(0, len(weights), GIF_STEP))

    def update(i):
        e = epochs[i]
        ax.clear()
        draw_frame(ax, frame_state(weights, biases, e), grid, shape, X, y)
        ax.set_title(f"Epoch {e}", fontsize=12)

    anim = FuncAnimation(fig, update, frames=len(epochs), interval=125)
    path = OUT / "learning_replay.gif"
    anim.save(str(path), writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    weights, biases, epoch, train, val = load_states()
    _, grid, shape, X, y = build_probe_space()
    print("states:", weights.shape, "grid:", grid.shape, "shape:", shape)
    print("still:", make_still(weights, biases, grid, shape, X, y, epoch, train, val))
    print("gif:", make_gif(weights, biases, grid, shape, X, y))


if __name__ == "__main__":
    main()
