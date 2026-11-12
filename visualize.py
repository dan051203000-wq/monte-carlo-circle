#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把蒙特卡罗估算圆的过程画出来：4 张图，对应 4 个不同的理解角度。

    图 1  scatter        —— 随机点长什么样，圆内/圆外怎么分色
    图 2  convergence    —— 投点数变多，估计值如何收敛
    图 3  distribution   —— 同一个 N 重复多次，估计值呈什么分布
    图 4  error_decay    —— 误差随 N 衰减的斜率，验证 1/√N 规律

运行：python visualize.py
输出：results/01_scatter.png ... results/04_error_decay.png
"""

from __future__ import annotations

import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from monte_carlo_circle import (
    PI_TRUE,
    convergence_curve,
    estimate_pi,
    random_points,
    repeat_experiment,
    theoretical_sigma,
)

OUT = "results"
INSIDE_COLOR = "#d94f4f"   # 圆内：红
OUTSIDE_COLOR = "#9aa5b1"  # 圆外：灰
LINE_COLOR = "#2f6fb2"     # 主曲线：蓝
TRUE_COLOR = "#2f9e6b"     # 真值参考线：绿
GRID_COLOR = "#e3e8ef"


def setup_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c7cfda",
        "axes.labelcolor": "#1f2933",
        "text.color": "#1f2933",
        "xtick.color": "#52606d",
        "ytick.color": "#52606d",
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
    })


# ---------------------------------------------------------------- 图 1：散点
def fig_scatter(n: int = 3000, radius: float = 1.0, seed: int = 7) -> str:
    rng = random.Random(seed)
    xs_in, ys_in, xs_out, ys_out = [], [], [], []
    r2 = radius * radius
    for x, y in random_points(n, radius, rng):
        if x * x + y * y <= r2:
            xs_in.append(x)
            ys_in.append(y)
        else:
            xs_out.append(x)
            ys_out.append(y)

    pi_hat = 4.0 * len(xs_in) / n
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.add_patch(Rectangle((-radius, -radius), 2 * radius, 2 * radius,
                           fill=False, lw=1.6, ec="#52606d"))
    ax.add_patch(Circle((0, 0), radius, fill=False, lw=2.4, ec=LINE_COLOR))
    ax.scatter(xs_out, ys_out, s=7, c=OUTSIDE_COLOR, alpha=0.55,
               linewidths=0, label=f"圆外 {len(xs_out):,} 个")
    ax.scatter(xs_in, ys_in, s=7, c=INSIDE_COLOR, alpha=0.7,
               linewidths=0, label=f"圆内 {len(xs_in):,} 个")

    ax.set_aspect("equal")
    ax.set_xlim(-radius * 1.08, radius * 1.08)
    ax.set_ylim(-radius * 1.08, radius * 1.08)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"随机撒 {n:,} 个点：落在圆内的比例 ≈ 圆面积 / 正方形面积 = π/4\n"
        f"推算 π ≈ 4k/N = {pi_hat:.4f}（真值 {PI_TRUE:.4f}，偏差 {pi_hat - PI_TRUE:+.4f}）",
        fontsize=11,
    )
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "01_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------ 图 2：收敛曲线
def fig_convergence(repeats: int = 5, seed: int = 2026) -> str:
    ns = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    rng = random.Random(seed)
    runs = []
    for n in ns:
        runs.append([estimate_pi(n, 1.0, rng).pi_hat for _ in range(repeats)])
    means = [sum(r) / len(r) for r in runs]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i in range(repeats):
        ax.plot(ns, [r[i] for r in runs], color=LINE_COLOR, lw=0.9, alpha=0.35,
                marker="o", ms=3, zorder=2)
    ax.plot(ns, means, color=LINE_COLOR, lw=2.2, marker="o", ms=5,
            label=f"{repeats} 次重复的均值", zorder=3)
    ax.axhline(PI_TRUE, color=TRUE_COLOR, lw=1.8, ls="--", label=f"真值 π = {PI_TRUE:.6f}", zorder=1)

    # 用理论 ±2σ 画出"合理波动带"
    upper = [PI_TRUE + 2 * theoretical_sigma(n) for n in ns]
    lower = [PI_TRUE - 2 * theoretical_sigma(n) for n in ns]
    ax.fill_between(ns, lower, upper, color=TRUE_COLOR, alpha=0.12,
                    label="理论波动带 ±2σ（σ = 1.6415/√N）")

    ax.set_xscale("log")
    ax.set_xlabel("投点数 N（对数轴）")
    ax.set_ylabel("π 的估计值")
    ax.set_title("收敛过程：点数越多，估计值越贴着真值，但带有随机抖动")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(PI_TRUE - 0.55, PI_TRUE + 0.55)
    fig.tight_layout()
    path = os.path.join(OUT, "02_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------ 图 3：分布形态
def fig_distribution(n: int = 5000, repeat: int = 1000, seed: int = 11) -> str:
    stat = repeat_experiment(n, repeat, 1.0, seed=seed)
    samples = stat["pi_samples"]
    mean, std = stat["pi_mean"], stat["pi_std"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(samples, bins=34, color=LINE_COLOR, alpha=0.75, edgecolor="white", linewidth=0.6,
            density=True, label=f"{repeat:,} 次独立实验（每次 N={n:,}）")

    # 叠加同样均值/标准差的正态曲线，说明估计量近似服从正态分布
    xs = [mean - 4 * std + 8 * std * i / 300 for i in range(301)]
    pdf = [math.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi)) for x in xs]
    ax.plot(xs, pdf, color="#7a3ea3", lw=2, label=f"正态分布 N({mean:.4f}, {std:.4f}²)")
    ax.axvline(PI_TRUE, color=TRUE_COLOR, lw=1.8, ls="--", label=f"真值 π = {PI_TRUE:.4f}")

    ax.set_xlabel("单次实验得到的 π 估计值")
    ax.set_ylabel("概率密度")
    ax.set_title("单次结果不可靠：上千次重复后，估计值呈钟形分布，均值逼近真值且无系统性偏差")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "03_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# -------------------------------------------------------- 图 4：误差衰减斜率
def fig_error_decay(repeats: int = 30, seed: int = 4242) -> str:
    """用均方根误差(RMSE)而不是单次绝对误差：单次误差随机性太大，画出来是锯齿。

    横纵都用对数轴后，若误差 ∝ N^(-1/2)，图像应是一条斜率 -0.5 的直线。
    这里顺手做一次最小二乘拟合，把实测斜率标在图上 —— 它就是"1/√N 规律"的实验证据。
    """
    ns = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    rng = random.Random(seed)
    rmse, sigma_th = [], []
    for n in ns:
        errs = [estimate_pi(n, 1.0, rng).pi_hat - PI_TRUE for _ in range(repeats)]
        rmse.append(math.sqrt(sum(e * e for e in errs) / len(errs)))
        sigma_th.append(theoretical_sigma(n))

    # 在 log-log 空间做一元线性回归，得到实测斜率
    lx = [math.log(n) for n in ns]
    ly = [math.log(e) for e in rmse]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum((x - mx) ** 2 for x in lx)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, rmse, color=LINE_COLOR, marker="o", lw=2,
            label=f"实测 RMSE（每档 {repeats} 次重复）")
    ax.plot(ns, sigma_th, color="#e08b2d", ls="--", lw=1.8,
            label="理论 σ = 1.6415/√N")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("投点数 N（对数轴）")
    ax.set_ylabel("π 估计值的均方根误差（对数轴）")
    ax.set_title("蒙特卡罗的效率天花板：误差只按 1/√N 下降\n"
                 f"实测拟合斜率 = {slope:.3f}（理论 -0.500）；点数翻 100 倍，误差才降 10 倍")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, "04_error_decay.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    setup_style()
    for fn in (fig_scatter, fig_convergence, fig_distribution, fig_error_decay):
        print("生成", fn.__name__, "->", fn())


if __name__ == "__main__":
    main()
