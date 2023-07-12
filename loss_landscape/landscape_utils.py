#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import copy
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D

from utils.common_utils import unwrap_model_fn

# https://github.com/xxxnell/how-do-vits-work


def rand_basis(ws: Dict, device: Optional[str] = torch.device("cpu")):
    return {k: torch.randn(size=v.shape, device=device) for k, v in ws.items()}


def normalize_filter(bs: Dict, ws: Dict):
    bs = {k: v.float() for k, v in bs.items()}
    ws = {k: v.float() for k, v in ws.items()}

    norm_bs = {}
    for k in bs:
        ws_norm = torch.norm(ws[k], dim=0, keepdim=True)
        bs_norm = torch.norm(bs[k], dim=0, keepdim=True)
        norm_bs[k] = ws_norm / (bs_norm + 1e-7) * bs[k]

    return norm_bs


def ignore_bn(ws: Dict):
    ignored_ws = {}
    for k in ws:
        if len(ws[k].size()) < 2:
            ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
        else:
            ignored_ws[k] = ws[k]
    return ignored_ws


def create_bases(
    model: torch.nn.Module,
    device: Optional[str] = torch.device("cpu"),
    has_module: Optional[bool] = False,
):
    unwrapped_model = unwrap_model_fn(model)
    weight_state_0 = unwrapped_model.state_dict()
    bases = [rand_basis(weight_state_0, device) for _ in range(2)]  # Use two bases
    bases = [normalize_filter(bs, weight_state_0) for bs in bases]
    bases = [ignore_bn(bs) for bs in bases]

    return bases


def generate_plots(xx, yy, zz, model_name, results_loc):
    zz = np.log(zz)

    plt.figure(figsize=(10, 10))
    plt.contour(xx, yy, zz)
    plt.savefig(f"{results_loc}/{model_name}_log_contour.png", dpi=100)
    plt.close()

    ## 3D plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_axis_off()
    surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.savefig(
        f"{results_loc}/{model_name}_log_surface.png",
        dpi=100,
        format="png",
        bbox_inches="tight",
    )
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.set_axis_off()

    def init():
        ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return (fig,)

    def animate(i):
        ax.view_init(elev=(15 * (i // 15) + i % 15) + 0.0, azim=i)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return (fig,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=100, interval=20, blit=True
    )

    anim.save(
        f"{results_loc}/{model_name}_log_surface.gif", fps=15, writer="imagemagick"
    )


def plot_save_graphs(
    save_dir: str,
    model_name: str,
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    loss_surface: np.ndarray,
    resolution: int,
):
    np.save(f"{save_dir}/{model_name}_xx.npy", grid_a)
    np.save(f"{save_dir}/{model_name}_yy.npy", grid_b)
    np.save(f"{save_dir}/{model_name}_zz.npy", loss_surface)

    plt.figure(figsize=(10, 10))
    plt.contour(grid_a, grid_b, loss_surface)
    plt.savefig(f"{save_dir}/{model_name}_contour_res_{resolution}.png", dpi=100)
    plt.close()

    generate_plots(
        xx=grid_a,
        yy=grid_b,
        zz=loss_surface,
        model_name=model_name,
        results_loc=save_dir,
    )
