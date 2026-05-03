import math
from typing import Literal

from scipy import stats


def proportion_diff_test(
    nA: int,
    cA: int,
    nB: int,
    cB: int,
    alternative: Literal["two-sided", "greater", "smaller"],
):
    """比率の差の検定

    Args:
        nA: A群のサンプルサイズ
        cA: A群のイベント数
        nB: B群のサンプルサイズ
        cB: B群のイベント数
        alternative: 対立仮説

    Returns:
        (A群とB群の比率の差, p値)
    """
    pA = cA / nA
    pB = cB / nB
    delta = pB - pA
    p = (cA + cB) / (nA + nB)
    se = math.sqrt(p * (1 - p) * (1 / nA + 1 / nB))
    z = delta / se

    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p = stats.norm.sf(z)
    elif alternative == "smaller":
        p = 1 - stats.norm.sf(z)

    return delta, p


if __name__ == "__main__":
    nA = 100000
    cA = 10000
    nB = 100000
    cB = math.ceil(nB * (cA / nA) * 1.03)

    # 両側検定の場合
    delta, p = proportion_diff_test(nA, cA, nB, cB, "two-sided")
    print(f"両側検定の場合: delta={delta}, p={p}")

    # pA < pBを対立仮説とした片側検定の場合
    delta, p = proportion_diff_test(nA, cA, nB, cB, "greater")
    print(f"pA < pBを対立仮説とした片側検定の場合: delta={delta}, p={p}")

    # pA > pBを対立仮説とした片側検定の場合
    delta, p = proportion_diff_test(nA, cA, nB, cB, "smaller")
    print(f"pA > pBを対立仮説とした片側検定の場合: delta={delta}, p={p}")
