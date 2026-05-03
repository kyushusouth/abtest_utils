import math
from statistics import NormalDist
from typing import Literal


def poisson_test_sample_size(
    lambdaA: float,
    lambdaB: float,
    alternative: Literal["two-sided", "one-sided"],
    alpha: float = 0.05,
    power: float = 0.8,
    alloc_ratio: float = 1.0,
) -> tuple[int, int]:
    """ポアソン分布の差の検定において必要なサンプルサイズを計算する

    Args:
        lambdaA: A群の平均イベント数
        lambdaB: B群の平均イベント数
        alpha: 有意水準
        power: 検出力
        alloc_ratio: A群とB群のサンプルサイズの比 (`nB / nA`)
        alternative: 対立仮説

    Returns:
        nA: A群のサンプルサイズ
        nB: B群のサンプルサイズ
    """
    if alternative == "two-sided":
        z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    elif alternative == "one-sided":
        z_alpha = NormalDist().inv_cdf(1 - alpha)

    z_beta = NormalDist().inv_cdf(power)

    delta = lambdaB - lambdaA
    var = lambdaA + lambdaB / alloc_ratio

    nA = ((z_alpha + z_beta) ** 2 * var) / (delta**2)
    nB = nA * alloc_ratio
    return math.ceil(nA), math.ceil(nB)


lambdaA = 0.5
lambdaB = lambdaA * 1.02

# 両側検定の場合
nA, nB = poisson_test_sample_size(lambdaA, lambdaB, "two-sided")
print(f"両側検定に必要なサンプルサイズ: A群={nA}, B群={nB}")

# 片側検定の場合
nA, nB = poisson_test_sample_size(lambdaA, lambdaB, "one-sided")
print(f"片側検定に必要なサンプルサイズ: A群={nA}, B群={nB}")
