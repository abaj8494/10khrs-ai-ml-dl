"""
Universal Approximation Demo – Mega‑Set Edition
==============================================
Generates a looping GIF that shows a tiny ReLU MLP learning **nine** very different
2‑D surfaces:

1. Mexican Hat     (smooth, oscillatory)
2. Rippling Sinc    (smooth, decaying)
3. Cylinder       (flat‑top, steep walls)
4. Letter “A”     (boolean pattern)
5. Windmill       (rotational sign pattern)
6. Paper Plane     (angled sliver)
7. Stairs        (step function)
8. Pyramid       (|x|+|y| ridge)
9. Green Cross     (axis‑aligned notches)

The script renders three side‑by‑side panes for each training snapshot:

* **Left:**   Model architecture + live metrics
* **Middle:** Ground‑truth surface (function name as title)
* **Right:**  Network prediction evolving toward the truth

Running the script
------------------
```bash
pip install torch matplotlib imageio tqdm
python universal_approx_animation.py   # → universal.gif (~768×256 px)
```
Embed with
```markdown
![Universal Approximation demo](universal.gif)
```
The GIF autoloops on GitHub (`loop=0`).
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Callable, List

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – 3‑D projection hook
from tqdm import tqdm

# ───────────────────────── CONFIGURABLE PARAMS ──────────────────────────── #
GIF_PATH = Path("universal.gif")
FRAME_DIR = Path("frames_tmp")
GRID_N = 50            # 50×50 grid ⇒ 2 500 samples per surface
SNAPSHOT_EVERY = 25    # capture a frame every N optimisation steps
EPOCHS_PER_FUNC = 400
DEVICE = torch.device("cpu")            # swap to "cuda" if you have a GPU handy
SEED = 42

CMAP_PRED = plt.cm.viridis
CMAP_TRUE = plt.cm.plasma

# ─────────────────────────────────────────────────────────────────────────── #

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────── TARGET TEST FUNCTIONS ──────────────────────────── #

def mexican_hat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r2 = x ** 2 + y ** 2
    return torch.sin(r2 * math.pi) / (1 + 5 * r2)


def ripple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(x ** 2 + y ** 2) + 1e-9
    return torch.sin(8 * r) / (8 * r)


def cylinder(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(x ** 2 + y ** 2)
    return torch.sigmoid(-40 * (r - 0.5))  # ≈1 inside, ≈0 outside

# The six glyph/shape surfaces supplied by the user ------------------------ #

tsign = torch.sign  # shorthand

def letter_A(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    term1 = (1 - tsign(-x - 0.9 + torch.abs(y * 2))) / 3 * (tsign(0.9 - x) + 1) / 3 * (tsign(x + 0.65) + 1) / 2
    term2 = (1 - tsign(-x - 0.39 + torch.abs(y * 2))) / 3 * (tsign(0.9 - x) + 1) / 3
    term3 = (1 - tsign(-x - 0.39 + torch.abs(y * 2))) / 3 * (tsign(0.6 - x) + 1) / 3 * (tsign(x - 0.35) + 1) / 2
    return term1 - term2 + term3


def windmill(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return tsign(x * y) * tsign(1 - (x * 9) ** 2 + (y * 9) ** 2) / 9


def paper_plane(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return tsign(x) * torch.atan(x * 80) / 6 * tsign(-y - x + 1) * tsign(-y + x + 1) * 5 - 1.01


def stairs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        tsign(-0.65 - x)
        + tsign(-0.35 - x)
        + tsign(-0.05 - x)
        + tsign(0.25 - x)
        + tsign(0.55 - x)
    ) / 7


def pyramid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1 - torch.abs(x + y) - torch.abs(y - x)


def green_cross(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    inner = tsign((x * 12) ** 2 - 9) - 1 + tsign((y * 12) ** 2 - 9) - 1
    return 0.1 - tsign(inner) / 2

# Assemble list & names ----------------------------------------------------- #
FUNCS: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = [
    mexican_hat,
    ripple,
    cylinder,
    letter_A,
    windmill,
    paper_plane,
    stairs,
    pyramid,
    green_cross,
]

FUNC_NAMES = [
    "Mexican Hat",
    "Rippling Sinc",
    "Cylinder",
    "Letter A",
    "Windmill",
    "Paper Plane",
    "Stairs",
    "Pyramid",
    "Green Cross",
]

# ───────────────────────────── NETWORK MODEL ────────────────────────────── #

def make_mlp(in_dim: int = 2, hidden: int = 64, depth: int = 2, out_dim: int = 1) -> nn.Module:
    layers: list[nn.Module] = []
    for i in range(depth):
        layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)

# ───────────────────────────── PLOTTING UTIL ────────────────────────────── #

def surface_plot(ax: Axes3D, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, *, cmap, title: str):
    ax.plot_surface(X, Y, Z, cmap=cmap, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=8, pad=8)


def capture_frame(epoch: int, func_idx: int, X: np.ndarray, Y: np.ndarray,
                  Z_true: np.ndarray, Z_pred: np.ndarray, loss: float,
                  param_text: str):
    fig = plt.figure(figsize=(7.68, 2.56), dpi=100)
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.axis("off")
    ax0.text(0, 1, param_text, va="top", ha="left", fontsize=8, family="monospace")

    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    surface_plot(ax1, X, Y, Z_true, cmap=CMAP_TRUE, title=FUNC_NAMES[func_idx])

    ax2 = fig.add_subplot(gs[0, 2], projection="3d")
    surface_plot(
        ax2, X, Y, Z_pred, cmap=CMAP_PRED,
        title=f"MLP Approx\nepoch {epoch}  loss {loss:.1e}")

    fig.savefig(FRAME_DIR / f"frame_{func_idx:02d}_{epoch:04d}.png", dpi=100)
    plt.close(fig)

# ─────────────────────────────── MAIN ────────────────────────────────────── #

def main():
    FRAME_DIR.mkdir(exist_ok=True)

    grid = torch.linspace(-1, 1, GRID_N)
    X_grid, Y_grid = torch.meshgrid(grid, grid, indexing="xy")
    XY = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1).to(DEVICE)

    model = make_mlp().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    arch_text_static = f"MLP 2→64→64→1\nParams: {total_params:,}\n"

    criterion = nn.MSELoss()

    for f_idx, f in enumerate(FUNCS):
        with torch.no_grad():
            Z_true = f(X_grid, Y_grid).detach().cpu().numpy()

        model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        opt = torch.optim.Adam(model.parameters(), lr=2e-2)

        for epoch in tqdm(range(1, EPOCHS_PER_FUNC + 1), desc=f"Training {FUNC_NAMES[f_idx]}"):
            pred_flat = model(XY).squeeze()
            loss = criterion(pred_flat, f(XY[:, 0], XY[:, 1]))

            opt.zero_grad(); loss.backward(); opt.step()

            if epoch % SNAPSHOT_EVERY == 0 or epoch in {1, EPOCHS_PER_FUNC}:
                with torch.no_grad():
                    Z_pred = model(XY).detach().cpu().numpy().reshape(GRID_N, GRID_N)
                param_text = arch_text_static + f"Epoch: {epoch}/{EPOCHS_PER_FUNC}\nLoss : {loss:.2e}"
                capture_frame(
                    epoch, f_idx, X_grid.numpy(), Y_grid.numpy(),
                    Z_true, Z_pred, loss.item(), param_text)

    # assemble GIF
    print("\nEncoding GIF…")
    frames = [iio.imread(fp) for fp in sorted(FRAME_DIR.glob("frame_*.png"))]
    shapes = {f.shape for f in frames}
    assert len(shapes) == 1, "Frame size mismatch – check savefig parameters."
    iio.imwrite(GIF_PATH, frames, duration=100, loop=0)

    shutil.rmtree(FRAME_DIR)
    print(f"Done ✨  {GIF_PATH} ready for your README!")


if __name__ == "__main__":
    main()

