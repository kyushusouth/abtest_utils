import math

from scipy import stats


def welch_t_non_inferiority_test(
    nA: int,
    mA: float,
    vA: float,
    nB: int,
    mB: float,
    vB: float,
    delta_l: float,
):
    """Welchのt検定を用いた非劣性検定
    
    Args:
        nA: A群のサンプルサイズ
        mA: A群の標本平均
        vA: A群の不偏分散
        nB: B群のサンプルサイズ
        mB: B群の標本平均
        vB: B群の不偏分散
        delta_l: lower equivalence bound

    Returns:
        p値
    """
    delta = mB - mA
    se = math.sqrt(vA / nA + vB / nB)

    t_l = (delta - delta_l) / se

    df_num = (vA / nA + vB / nB) ** 2
    df_denom = (vA / nA) ** 2 / (nA - 1) + (vB / nB) ** 2 / (nB - 1)
    df = df_num / df_denom

    p_l = stats.t.sf(t_l, df)
    return p_l


if __name__ == "__main__":
    nA = 100000
    mA = 0.1
    vA = 0.1
    nB = 100000
    mB = mA * 1.01
    vB = vA
    delta_l = -(mB - mA) * 0.5

    p_l = welch_t_non_inferiority_test(nA, mA, vA, nB, mB, vB, delta_l)
    print(p_l)
