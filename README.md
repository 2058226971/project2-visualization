# Project 2 — Classifier Visualization

A reproducible implementation of the SZU Python course Project 2. The supplied
`main.py` prepares the Iris data, fits the models and drives the plotting; this
repo completes the four required visualization tasks plus one extension.

**Assessment note:** only the plotting code and the figures are written by the
student; the models and data pipeline come from the course starter.

## What this project does

Using the Iris dataset, we visualize **what fitted classifiers predict**:

- **Task 1** — compare logistic regression and a random forest on two petal
  features, by plotting their decision regions and per-species probabilities.
- **Task 2** — draw the 3D plane where a binary logistic model's
  P(virginica) = 0.5, using three sepal/petal features.
- **Task 3** — show how that probability varies across the whole 3D feature
  space with colored probe points.
- **Task 4 (extension)** — replay 301 saved training states to show how the
  decision region moves while the model learns, together with accuracy curves.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt` (numpy, matplotlib, scikit-learn,
  Pillow)

## How to reproduce

```bash
# 1. install dependencies
python -m pip install -r requirements.txt

# 2. generate all six core figures
python main.py

# 3. verify every required plotting call is implemented
python main.py --completed
#    -> prints "Unfinished plotting calls: 0" and exits with status 0

# 4. generate the Task 4 extension outputs
python extension.py
```

All figures are written to `outputs/`.

## Output files

| File | Source | Description |
| --- | --- | --- |
| `logistic_regions.png` | Task 1 | Logistic-regression decision regions + flowers |
| `logistic_probabilities.png` | Task 1 | P(setosa/versicolor/virginica) panels |
| `forest_regions.png` | Task 1 | Random-forest decision regions + flowers |
| `forest_probabilities.png` | Task 1 | Random-forest probability panels |
| `boundary_3d.png` | Task 2 | 0.5 decision plane with measured flowers |
| `probability_3d.png` | Task 3 | 3D probability probes for virginica |
| `learning_replay.png` | Task 4 | Epoch 0/15/300 maps + accuracy curves |
| `learning_replay.gif` | Task 4 | 31-frame animation of the full training run |

## Implementation notes

- **`tasks.py`** contains only plotting functions. Fitting is kept in
  `main.py`, so every figure is reproducible from the same inputs. This is
  deliberate and keeps the code easy to explain in the report.
- Class colors are consistent across all figures: setosa = coral,
  versicolor = green, virginica = blue. A species keeps the same color
  everywhere, so a reader can follow it from the 2D regions to the 3D views.
- All probability panels and the 3D colorbar use a fixed **0–1** scale;
  contour levels are set explicitly instead of being inferred from the data.
- **Validation setup:** the 2D comparison trains on 105 flowers and evaluates
  on a stratified 45-flower holdout (`random_state=42`). The logistic model
  reaches 91.1% (41/45), the random forest 95.6% (43/45). The 3D binary model
  is fitted on all 100 non-setosa flowers in raw centimetres, so it is a
  different experiment from the 2D comparison.
- **Task 4:** `extension.py` loads `weights`/`biases` from
  `data/learning_states.npz` and accuracy values from
  `data/learning_history.csv`. Nothing is retrained. The probe grid is
  standardized with the same scaler fitted on the training split, so only the
  parameters change from frame to frame.

## Repository layout

```
project2-visualization-student/
├── main.py                 # supplied pipeline (do not edit)
├── tasks.py                # four required plotting functions + draw_frame hook
├── extension.py            # Task 4 saved-state replay
├── requirements.txt
├── data/
│   ├── learning_history.csv
│   └── learning_states.npz
├── task-sources/           # lecture slides for Tasks 1–4
├── outputs/                # generated figures (this README's results)
└── report/                 # Overleaf-compatible LaTeX report
    ├── main.tex
    └── figs/               # copies of the generated figures
```

## Report

`report/main.tex` is built on the
[SZU Experiment Report Template](https://github.com/ChaoFan996/Experiment-Report-Template-SZU).
Upload the `report/` folder to Overleaf as a new project and compile with
**XeLaTeX**. Fill in your name, student ID and GitHub URL where marked in the
cover page before submitting.
