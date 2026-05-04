import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.hypothesis_test.non_inferiority import welch_t_non_inferiority_test
from src.sample_size.non_inferiority import welch_t_non_inferiority_test_sample_size


def main():
    project_dir = Path(__file__).parent.parent.parent
    result_dir = project_dir.joinpath("data", "hypothesis_test_simulation")
    result_dir.mkdir(parents=True, exist_ok=True)

    # パラメータの設定
    mA = 0.1
    vA = 0.1
    alpha = 0.05
    power = 0.8
    alloc_ratio = 1.0
    n_sim = 10000

    # テストケースの定義
    # mBはmAより少し劣化している(0.099)とする。真の差は -0.001
    test_cases = [
        {
            "name": "対立仮説が真である時に正しく棄却できる割合の確認",
            "is_allowable": True,
            "mB": mA * 0.99,
            "vB": vA,
        },
        {
            "name": "帰無仮説が真である時に誤って棄却してしまう割合の確認",
            "is_allowable": False,
            "mB": mA * 0.99,
            "vB": vA,
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"--- {test_case['name']} ---")

        true_diff = test_case["mB"] - mA

        # lower equivalence boundを定義
        if test_case["is_allowable"]:
            # 真の劣化幅よりも許容するマージンが広い状況（対立仮説が真である状況）
            delta_l = true_diff * 1.5
        else:
            # 真の劣化幅が許容するマージンを超えない状況（帰無仮説が真である状況）
            # true_diffに近すぎるとサンプルサイズが大きすぎて落ちるので余裕を持たせる
            delta_l = true_diff * 0.5

        print(f"真の差 = {true_diff:.4f}")
        print(f"delta_l = {delta_l:.4}")

        # サンプルサイズの計算
        nA, nB = welch_t_non_inferiority_test_sample_size(
            mA,
            test_case["mB"],
            vA,
            test_case["vB"],
            delta_l,
            alpha,
            power,
            alloc_ratio,
        )
        print(f"必要サンプルサイズ: nA = {nA}, nB = {nB}")

        pvalues = np.zeros(n_sim)
        for i in tqdm(range(n_sim)):
            rng = np.random.default_rng(i)
            dataA = rng.normal(mA, math.sqrt(vA), nA)
            dataB = rng.normal(test_case["mB"], math.sqrt(test_case["vB"]), nB)

            p = welch_t_non_inferiority_test(
                nA,
                np.mean(dataA),
                np.var(dataA, ddof=1),
                nB,
                np.mean(dataB),
                np.var(dataB, ddof=1),
                delta_l,
            )
            pvalues[i] = p

        if test_case["is_allowable"]:
            empirical_power = np.mean(pvalues < alpha)
            print(
                f"帰無仮説を正しく棄却できた割合 (期待値 ~{power}): {empirical_power:.4f}"
            )
            result_dict = {
                "test_case_name": test_case["name"],
                "is_allowable": test_case["is_allowable"],
                "sample_size_A": int(nA),
                "sample_size_B": int(nB),
                "empirical_power": float(empirical_power),
            }
        else:
            type_i_error_rate = np.mean(pvalues < alpha)
            print(
                f"帰無仮説を誤って棄却した割合 (期待値 ~0.0): {type_i_error_rate:.4f}"
            )
            result_dict = {
                "test_case_name": test_case["name"],
                "is_allowable": test_case["is_allowable"],
                "sample_size_A": int(nA),
                "sample_size_B": int(nB),
                "type_i_error_rate": float(type_i_error_rate),
            }

        print()
        results.append(result_dict)

    output_path = result_dir.joinpath("non_inferiority.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
