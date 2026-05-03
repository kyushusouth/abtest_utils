import math
from statistics import NormalDist


def welch_t_equivalence_test_sample_size(
    mA: float,
    mB: float,
    vA: float,
    vB: float,
    delta_l: float,
    delta_u: float,
    alpha: float = 0.05,
    power: float = 0.8,
    alloc_ratio: float = 1.0,
):
    """Welchのt検定を用いた同等性検定において必要なサンプルサイズを計算する
    
    Args:
        mA: A群の標本平均
        mB: B群の標本平均
        vA: A群の不偏分散
        vB: B群の不偏分散
        delta_l: lower equivalence bound
        delta_u: upper equivalence bound 
        alpha: 有意水準
        power: 検出力
        alloc_ratio: A群とB群のサンプルサイズの比 (`nB / nA`)

    Returns:
        nA: A群のサンプルサイズ
        nB: B群のサンプルサイズ
    """
    z_alpha = NormalDist().inv_cdf(1 - alpha)
    z_beta = NormalDist().inv_cdf(power)

    delta = mB - mA
    
    nA_l = (z_alpha + z_beta) ** 2 * (vA + vB / alloc_ratio) / (delta - delta_l) ** 2
    nB_l = nA_l * alloc_ratio
    
    nA_u = (z_alpha + z_beta) ** 2 * (vA + vB / alloc_ratio) / (delta - delta_u) ** 2
    nB_u = nA_u * alloc_ratio
    
    nA = max(nA_l, nA_u)
    nB = max(nB_l, nB_u)
    return math.ceil(nA), math.ceil(nB)


mA = 0.1
mB = mA * 1.009
vA = 0.1
vB = vA
delta_l = -0.001
delta_u = 0.001

nA, nB = welch_t_equivalence_test_sample_size(mA, mB, vA, vB, delta_l, delta_u)
print(f"必要なサンプルサイズ: A群={nA}, B群={nB}")
