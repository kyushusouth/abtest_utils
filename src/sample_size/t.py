import math
from statistics import NormalDist
from typing import Literal


def welch_t_test_sample_size(
    mA: float,
    mB: float,
    vA: float,
    vB: float,
    alternative: Literal["two-sided", "one-sided"],
    alpha: float = 0.05,
    power: float = 0.8,
    alloc_ratio: float = 1.0,
) -> tuple[int, int]:
    """Welchのt検定において必要なサンプルサイズを計算する

    Args:
        mA: A群の標本平均
        mB: B群の標本平均
        vA: A群の不偏分散
        vB: B群の不偏分散
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

    delta = mB - mA
    nA = (z_alpha + z_beta) ** 2 * (vA + vB / alloc_ratio) / delta**2
    nB = nA * alloc_ratio
    return math.ceil(nA), math.ceil(nB)


if __name__ == "__main__":
    mA = 0.1
    mB = mA * 1.03
    vA = 0.1
    vB = vA

    # 両側検定の場合
    nA, nB = welch_t_test_sample_size(mA, mB, vA, vB, "two-sided")
    print(f"両側検定に必要なサンプルサイズ: A群={nA}, B群={nB}")

    # 片側検定の場合
    nA, nB = welch_t_test_sample_size(mA, mB, vA, vB, "one-sided")
    print(f"片側検定に必要なサンプルサイズ: A群={nA}, B群={nB}")
