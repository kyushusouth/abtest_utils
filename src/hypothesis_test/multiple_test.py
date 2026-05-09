def bonferroni_correction(
    p_values: list[float], alpha: float
) -> tuple[list[float], list[bool]]:
    """ボンフェローニ法による多重比較の補正

    Args:
        p_values: 各検定のp値のリスト
        alpha: 有意水準

    Returns:
        (補正後のp値のリスト, 有意差ありかどうかのリスト)（元のリストと同じ順序）
    """
    m = len(p_values)
    adjusted = [min(p * m, 1.0) for p in p_values]
    rejected = [p <= alpha for p in adjusted]
    return adjusted, rejected


def holm_correction(
    p_values: list[float], alpha: float
) -> tuple[list[float], list[bool]]:
    """Holm法による多重比較の補正

    Args:
        p_values: 各検定のp値のリスト
        alpha: 有意水準

    Returns:
        (補正後のp値のリスト, 有意差ありかどうかのリスト)（元のリストと同じ順序）
    """
    m = len(p_values)

    # p値の小さい順にargsort
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    rejected = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        adjusted[orig_idx] = min(adj, 1.0)

        # 棄却できない帰無仮説の場合は処理を終える
        if adj <= alpha:
            rejected[orig_idx] = True
        else:
            break
    return adjusted, rejected


def benjamini_hochberg_correction(
    p_values: list[float], alpha: float
) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg法による多重比較の補正

    Args:
        p_values: 各検定のp値のリスト
        alpha: 有意水準

    Returns:
        (補正後のp値のリスト, 有意差ありかどうかのリスト)（元のリストと同じ順序）
    """
    m = len(p_values)

    # p値の小さい順にargsort
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # p_(k) <= alpha * k/m を満たす最大のkを求める
    last_rejected_rank = -1
    for rank, (_, p) in enumerate(indexed):
        if p <= alpha * (rank + 1) / m:
            last_rejected_rank = rank

    # k <= last_rejected_rankを満たす全ての帰無仮説を棄却する
    adjusted = [0.0] * m
    rejected = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(p * m / (rank + 1), 1.0)
        rejected[orig_idx] = rank <= last_rejected_rank
    return adjusted, rejected


if __name__ == "__main__":
    p_values = [0.001, 0.008, 0.029, 0.041, 0.210]
    alpha = 0.05

    print(f"元のp値: {p_values}")

    adj, rej = bonferroni_correction(p_values, alpha)
    print(f"Bonferroni補正後:         p={[round(p, 4) for p in adj]}, 有意差={rej}")

    adj, rej = holm_correction(p_values, alpha)
    print(f"Holm補正後:               p={[round(p, 4) for p in adj]}, 有意差={rej}")

    adj, rej = benjamini_hochberg_correction(p_values, alpha)
    print(f"Benjamini-Hochberg補正後: p={[round(p, 4) for p in adj]}, 有意差={rej}")
