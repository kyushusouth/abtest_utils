import math
from typing import Literal

from scipy import stats


def welch_t_test(
    nA: int,
    mA: float,
    vA: float,
    nB: int,
    mB: float,
    vB: float,
    alternative: Literal["two-sided", "greater", "smaller"],
):
    """Welchのt検定

    Args:
        nA: A群のサンプルサイズ
        mA: A群の標本平均
        vA: A群の不偏分散
        nB: B群のサンプルサイズ
        mB: B群の標本平均
        vB: B群の不偏分散
        alternative: 対立仮説

    Returns:
        (A群とB群の標本平均の差, p値)
    """
    delta = mB - mA
    se = math.sqrt(vA / nA + vB / nB)
    t = delta / se

    df_num = (vA / nA + vB / nB) ** 2
    df_denom = (vA / nA) ** 2 / (nA - 1) + (vB / nB) ** 2 / (nB - 1)
    df = df_num / df_denom

    if alternative == "two-sided":
        p = 2 * stats.t.sf(abs(t), df)
    elif alternative == "greater":
        p = stats.t.sf(t, df)
    elif alternative == "smaller":
        p = 1 - stats.t.sf(t, df)

    return delta, p


def welch_t_test_cuped(
    nA: int,
    mA: float,
    mA_pre: float,
    vA: float,
    nB: int,
    mB: float,
    mB_pre: float,
    vB: float,
    rho: float,
    alternative: Literal["two-sided", "greater", "smaller"],
):
    """CUPEDを用いたWelchのt検定

    Args:
        nA: A群のサンプルサイズ
        mA: A群の標本平均
        mA_pre: A群の実験前データから計算した標本平均
        vA: A群の不偏分散
        nB: B群のサンプルサイズ
        mB: B群の標本平均
        mB_pre: B群の実験前データから計算した標本平均
        vB: B群の不偏分散
        rho: 共変量X,Yの相関係数
        alternative: 対立仮説

    Returns:
        (CUPEDにおけるΔ, p値)
    """
    delta = (mB - mA) - rho * (mB_pre - mA_pre)
    se = math.sqrt((vA / nA + vB / nB) * (1 - rho**2))
    t = delta / se

    df_num = (vA / nA + vB / nB) ** 2
    df_denom = (vA / nA) ** 2 / (nA - 1) + (vB / nB) ** 2 / (nB - 1)
    df = df_num / df_denom

    if alternative == "two-sided":
        p = 2 * stats.t.sf(abs(t), df)
    elif alternative == "greater":
        p = stats.t.sf(t, df)
    elif alternative == "smaller":
        p = 1 - stats.t.sf(t, df)

    return delta, p


if __name__ == "__main__":
    nA = 100000
    mA = 0.1
    vA = 0.1
    nB = 100000
    mB = mA * 1.03
    vB = vA

    # 両側検定の場合
    delta, p = welch_t_test(nA, mA, vA, nB, mB, vB, "two-sided")
    print(f"両側検定の場合: delta={delta}, p={p}")

    # mA < mBを対立仮説とした片側検定の場合
    delta, p = welch_t_test(nA, mA, vA, nB, mB, vB, "greater")
    print(f"mA < mBを対立仮説とした片側検定の場合: delta={delta}, p={p}")

    # mA > mBを対立仮説とした片側検定の場合
    delta, p = welch_t_test(nA, mA, vA, nB, mB, vB, "smaller")
    print(f"mA > mBを対立仮説とした片側検定の場合: delta={delta}, p={p}")
