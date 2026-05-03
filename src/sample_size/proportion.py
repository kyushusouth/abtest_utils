import math
from statistics import NormalDist
from typing import Literal


def proportion_diff_test_sample_size(
    pA: float,
    pB: float,
    alternative: Literal["two-sided", "one-sided"],
    alpha: float = 0.05,
    power: float = 0.8,
    alloc_ratio: float = 1.0,
) -> tuple[int, int]:
    """比率の差の検定において必要なサンプルサイズを計算する

    Args:
        pA: A群の比率
        pB: B群の比率
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

    p = (pA + alloc_ratio * pB) / (1 + alloc_ratio)
    numerator1 = z_alpha * math.sqrt((alloc_ratio + 1) * p * (1 - p))
    numerator2 = z_beta * math.sqrt(alloc_ratio * pA * (1 - pA) + pB * (1 - pB))
    numerator = (numerator1 + numerator2) ** 2
    denominator = alloc_ratio * (pA - pB) ** 2
    nA = numerator / denominator
    nB = alloc_ratio * nA
    return math.ceil(nA), math.ceil(nB)


if __name__ == "__main__":
    pA = 0.02
    pB = pA * 1.06

    # 両側検定の場合
    nA, nB = proportion_diff_test_sample_size(pA, pB, "two-sided")
    print(f"両側検定に必要なサンプルサイズ: A群={nA}, B群={nB}")

    # 片側検定の場合
    nA, nB = proportion_diff_test_sample_size(pA, pB, "one-sided")
    print(f"片側検定に必要なサンプルサイズ: A群={nA}, B群={nB}")
