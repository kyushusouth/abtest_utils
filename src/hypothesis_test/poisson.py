import math
from typing import Literal

from scipy import stats


def poisson_diff_test(
    nA: int,
    cA: int,
    nB: int,
    cB: int,
    alternative: Literal["two-sided", "greater", "smaller"],
):
    """Wald法によるポアソン分布の差の検定

    Args:
        nA: A群のサンプルサイズ
        cA: A群のイベント数
        nB: B群のサンプルサイズ
        cB: B群のイベント数
        alternative: 対立仮説

    Returns:
        (A群とB群の平均イベント数の差, p値)
    """
    lambdaA = cA / nA
    lambdaB = cB / nB
    delta = lambdaB - lambdaA
    se = math.sqrt(lambdaA / nA + lambdaB / nB)
    z = delta / se

    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p = stats.norm.sf(z)
    elif alternative == "smaller":
        p = 1 - stats.norm.sf(z)

    return delta, p


nA = 100000
cA = 10000
nB = 100000
cB = math.ceil(nB * (cA / nA) * 1.03)

# 両側検定の場合
delta, p = poisson_diff_test(nA, cA, nB, cB, "two-sided")
print(f"両側検定の場合: delta={delta}, p={p}")

# lambdaA < lambdaBを対立仮説とした片側検定の場合
delta, p = poisson_diff_test(nA, cA, nB, cB, "greater")
print(f"lambdaA < lambdaBを対立仮説とした片側検定の場合: delta={delta}, p={p}")

# lambdaA > lambdaBを対立仮説とした片側検定の場合
delta, p = poisson_diff_test(nA, cA, nB, cB, "smaller")
print(f"lambdaA > lambdaBを対立仮説とした片側検定の場合: delta={delta}, p={p}")
