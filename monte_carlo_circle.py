#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""蒙特卡罗方法"手搓"圆 —— 用随机投点估算圆面积与圆周率 π

核心思想（一句话）
------------------
在一个边长为 2r 的正方形里随机撒 N 个点，点落在内切圆（半径 r）内的概率
等于圆面积与正方形面积之比 πr² / 4r² = π/4。
于是只要数出落在圆内的点数 k，就能反推：π ≈ 4k/N，圆面积 S ≈ 4r²·k/N。

本模块全部使用标准库实现（random 模块提供均匀随机数），
不依赖 numpy / scipy，方便"手搓"过程被完整看清。

运行示例
--------
    python monte_carlo_circle.py                    # 默认 N=1_000_000，重复 10 次
    python monte_carlo_circle.py -n 100000 -r 2     # 半径 r=2 的圆，10 万点
    python monte_carlo_circle.py --repeat 200 --export
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Iterator, List, Tuple

# ---- 理论真值（用于对照，不参与估算）----
PI_TRUE = math.pi


# =========================================================================
# 1. 随机投点：真正的"手搓"部分
# =========================================================================
def random_points(n: int, radius: float = 1.0,
                  rng: random.Random | None = None) -> Iterator[Tuple[float, float]]:
    """在边长 2r、中心在原点的正方形内均匀生成 n 个随机点。

    均匀分布的做法：x、y 各自在 [-r, r] 上取 uniform。
    注意这一步不能写成 `r * (2 * rng.random() - 1)` 之外的花活 ——
    只要是独立同分布的均匀分布即可，但千万不要用"极坐标 + 均匀角度"，
    那样点会往圆心堆积，估算结果会系统性偏小（见 README 的常见错误）。
    """
    rng = rng or random.Random()
    uniform = rng.uniform
    for _ in range(n):
        yield uniform(-radius, radius), uniform(-radius, radius)


def count_inside(n: int, radius: float = 1.0,
                 rng: random.Random | None = None) -> int:
    """统计 n 个随机点中落在圆内的个数 k。

    判定条件就是圆的定义：x² + y² ≤ r²。
    为了省掉每次乘方，可以两边都不开方、直接比较 x²+y² 与 r²。
    """
    r2 = radius * radius
    k = 0
    for x, y in random_points(n, radius, rng):
        if x * x + y * y <= r2:
            k += 1
    return k


# =========================================================================
# 2. 一次估算 + 误差估计
# =========================================================================
@dataclass
class Trial:
    """单次实验的记录。"""
    n: int          # 投点数
    radius: float   # 半径
    k: int          # 圆内点数
    pi_hat: float   # π 的估计值 = 4k/n
    area_hat: float # 圆面积估计 = 4r²k/n

    @property
    def pi_error(self) -> float:
        return abs(self.pi_hat - PI_TRUE)

    @property
    def rel_error(self) -> float:
        return self.pi_error / PI_TRUE


def estimate_pi(n: int, radius: float = 1.0,
                rng: random.Random | None = None) -> Trial:
    """一次完整的蒙特卡罗估算。"""
    if n <= 0:
        raise ValueError("投点数 n 必须为正整数")
    if radius <= 0:
        raise ValueError("半径 radius 必须为正数")
    k = count_inside(n, radius, rng)
    pi_hat = 4.0 * k / n
    area_hat = 4.0 * radius * radius * k / n
    return Trial(n=n, radius=radius, k=k, pi_hat=pi_hat, area_hat=area_hat)


def theoretical_sigma(n: int) -> float:
    """π 估计量的理论标准差。

    k ~ B(n, p)，p = π/4，于是 Var(4k/n) = 16·p(1-p)/n，
    代入 p 得 σ_π = 4·sqrt(p(1-p)/n) ≈ 1.6415 / sqrt(n)。
    这条 1/√n 规律解释了为什么蒙特卡罗"精度每提高 10 倍，点数要涨 100 倍"。
    """
    p = PI_TRUE / 4.0
    return 4.0 * math.sqrt(p * (1.0 - p) / n)


# =========================================================================
# 3. 多批次重复实验：均值、标准差、置信区间
# =========================================================================
def repeat_experiment(n: int, repeat: int, radius: float = 1.0,
                      seed: int | None = None) -> dict:
    """独立重复 repeat 次，观察估计量的波动。

    这是蒙特卡罗方法最容易被忽略的一步：单次结果带偶然性，
    必须靠多次重复给出"区间"而不是"一个数字"。
    """
    rng = random.Random(seed)
    trials: List[Trial] = [estimate_pi(n, radius, rng) for _ in range(repeat)]
    pis = [t.pi_hat for t in trials]
    mean = sum(pis) / len(pis)
    var = sum((v - mean) ** 2 for v in pis) / (len(pis) - 1) if len(pis) > 1 else 0.0
    std = math.sqrt(var)
    stderr = std / math.sqrt(len(pis))
    return {
        "n_per_trial": n,
        "repeat": repeat,
        "radius": radius,
        "pi_mean": mean,
        "pi_std": std,
        "pi_stderr": stderr,
        "ci95": (mean - 1.959964 * stderr, mean + 1.959964 * stderr),
        "theoretical_sigma": theoretical_sigma(n),
        "bias": mean - PI_TRUE,
        "mean_abs_error": sum(t.pi_error for t in trials) / len(trials),
        "pi_samples": pis,
        "area_mean": sum(t.area_hat for t in trials) / len(trials),
        "area_true": PI_TRUE * radius * radius,
    }


# =========================================================================
# 4. 收敛过程：点数从少到多，看估计值如何"靠拢"
# =========================================================================
def convergence_curve(ns: List[int], repeats: int = 5,
                      seed: int = 20261112) -> List[dict]:
    """对一串点数规模分别做 repeats 次实验，返回每一步的统计信息。"""
    rng = random.Random(seed)
    rows = []
    for n in ns:
        sample = [estimate_pi(n, 1.0, rng).pi_hat for _ in range(repeats)]
        mean = sum(sample) / len(sample)
        rows.append({
            "n": n,
            "pi_mean": mean,
            "pi_min": min(sample),
            "pi_max": max(sample),
            "abs_error": abs(mean - PI_TRUE),
            "theoretical_sigma": theoretical_sigma(n),
        })
    return rows


# =========================================================================
# 5. 输出
# =========================================================================
def print_trial(t: Trial) -> None:
    print(f"  投点数 N       : {t.n:,}")
    print(f"  半径 r         : {t.radius}")
    print(f"  圆内点数 k     : {t.k:,}   (占比 {t.k / t.n:.6f}，理论值 π/4 = {PI_TRUE / 4:.6f})")
    print(f"  π 估计值       : {t.pi_hat:.6f}   真值 {PI_TRUE:.6f}   绝对误差 {t.pi_error:.6f}")
    print(f"  圆面积估计     : {t.area_hat:.6f}   真值 {PI_TRUE * t.radius ** 2:.6f}")


def export_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="蒙特卡罗方法估算圆面积与 π")
    parser.add_argument("-n", "--n", type=int, default=1_000_000, help="投点数（默认 1,000,000）")
    parser.add_argument("-r", "--radius", type=float, default=1.0, help="圆半径（默认 1.0）")
    parser.add_argument("--repeat", type=int, default=10, help="重复实验次数（默认 10）")
    parser.add_argument("--seed", type=int, default=20261112, help="随机种子，保证结果可复现")
    parser.add_argument("--export", action="store_true", help="把结果导出到 results/ 目录")
    args = parser.parse_args()

    print("=" * 62)
    print("蒙特卡罗方法 · 手搓圆形估算 π 与圆面积")
    print("=" * 62)

    t0 = time.perf_counter()
    trial = estimate_pi(args.n, args.radius, random.Random(args.seed))
    cost = time.perf_counter() - t0
    print(f"\n【单次实验】耗时 {cost:.2f}s")
    print_trial(trial)

    t0 = time.perf_counter()
    stat = repeat_experiment(args.n, args.repeat, args.radius, seed=args.seed + 1)
    cost2 = time.perf_counter() - t0
    lo, hi = stat["ci95"]
    print(f"\n【重复 {args.repeat} 次】耗时 {cost2:.2f}s（每次 {args.n:,} 点）")
    print(f"  π 估计均值     : {stat['pi_mean']:.6f}")
    print(f"  样本标准差     : {stat['pi_std']:.6f}   理论 σ = {stat['theoretical_sigma']:.6f}")
    print(f"  95% 置信区间   : [{lo:.6f}, {hi:.6f}]   是否包含真值 π：{'是' if lo <= PI_TRUE <= hi else '否'}")
    print(f"  平均绝对误差   : {stat['mean_abs_error']:.6f}")
    print(f"  圆面积均值     : {stat['area_mean']:.6f}   真值 {stat['area_true']:.6f}")

    # 精度 - 成本换算：这是蒙特卡罗最实用的结论
    print("\n【精度 vs 成本】π 的误差 ≈ 1.6415 / √N")
    for target in (1e-2, 1e-3, 1e-4, 1e-6):
        need = (1.6415 / target) ** 2
        print(f"  想要误差 ≤ {target:.0e}  →  需要约 {need:,.0f} 个点")

    if args.export:
        os.makedirs("results", exist_ok=True)
        ns = [100, 300, 1000, 3000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
        curve = convergence_curve(ns, repeats=args.repeat, seed=args.seed)
        export_csv("results/convergence.csv", curve,
                   ["n", "pi_mean", "pi_min", "pi_max", "abs_error", "theoretical_sigma"])
        summary = {k: v for k, v in stat.items() if k != "pi_samples"}
        summary["ci95"] = list(summary["ci95"])
        summary["single_trial"] = asdict(trial)
        with open("results/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n已导出：results/convergence.csv, results/summary.json")


if __name__ == "__main__":
    main()
