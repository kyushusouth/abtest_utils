import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.hypothesis_test.t import welch_t_test
from src.sample_size.t import welch_t_test_sample_size


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
    test_cases = [
        {
            "name": "正規分布に従うデータに対する両側検定",
            "dist": "normal",
            "alternative": "two-sided",
            "mB": mA * 1.02,
            "vB": vA,
        },
        {
            "name": "正規分布に従うデータに対する片側検定（A<B）",
            "dist": "normal",
            "alternative": "greater",
            "mB": mA * 1.02,
            "vB": vA,
        },
        {
            "name": "正規分布に従うデータに対する片側検定（A>B）",
            "dist": "normal",
            "alternative": "smaller",
            "mB": mA * 0.98,
            "vB": vA,
        },
        {
            "name": "ポアソン分布に従うデータに対する両側検定",
            "dist": "poisson",
            "alternative": "two-sided",
            "mB": mA * 1.02,
            "vB": vA,
        },
        {
            "name": "ポアソン分布に従うデータに対する片側検定（A<B）",
            "dist": "poisson",
            "alternative": "greater",
            "mB": mA * 1.02,
            "vB": vA,
        },
        {
            "name": "ポアソン分布に従うデータに対する片側検定（A>B）",
            "dist": "poisson",
            "alternative": "smaller",
            "mB": mA * 0.98,
            "vB": vA,
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"--- {test_case['name']} ---")

        # サンプルサイズの計算
        if test_case["alternative"] == "two-sided":
            nA, nB = welch_t_test_sample_size(
                mA,
                test_case["mB"],
                vA,
                test_case["vB"],
                test_case["alternative"],
                alpha,
                power,
                alloc_ratio,
            )
        elif test_case["alternative"] in ("greater", "smaller"):
            nA, nB = welch_t_test_sample_size(
                mA,
                test_case["mB"],
                vA,
                test_case["vB"],
                "one-sided",
                alpha,
                power,
                alloc_ratio,
            )
        else:
            raise ValueError("Not Implemented")

        print(f"必要サンプルサイズ: nA = {nA}, nB = {nB}")

        # 第一種の誤り発生確率がα程度であることを確認
        pvalues_aa = np.zeros(n_sim)
        for i in tqdm(range(n_sim)):
            rng = np.random.default_rng(i)

            # 同じ分布からデータを生成
            if test_case["dist"] == "normal":
                dataA = rng.normal(mA, math.sqrt(vA), nA)
                dataB = rng.normal(mA, math.sqrt(vA), nB)
            elif test_case["dist"] == "poisson":
                dataA = rng.poisson(mA, nA)
                dataB = rng.poisson(mA, nB)

            _, p = welch_t_test(
                nA,
                np.mean(dataA),
                np.var(dataA, ddof=1),
                nB,
                np.mean(dataB),
                np.var(dataB, ddof=1),
                test_case["alternative"],
            )
            pvalues_aa[i] = p

        type_i_error_rate = np.mean(pvalues_aa < alpha)
        print(
            f"帰無仮説を誤って棄却した割合 (期待値 ~{alpha}): {type_i_error_rate:.4f}"
        )

        # 検出力が(1-β)程度であることを確認
        pvalues_ab = np.zeros(n_sim)
        for i in tqdm(range(n_sim)):
            rng = np.random.default_rng(i)

            # 異なる分布からデータを生成
            if test_case["dist"] == "normal":
                dataA = rng.normal(mA, math.sqrt(vA), nA)
                dataB = rng.normal(test_case["mB"], math.sqrt(test_case["vB"]), nB)
            elif test_case["dist"] == "poisson":
                dataA = rng.poisson(mA, nA)
                dataB = rng.poisson(test_case["mB"], nB)

            _, p = welch_t_test(
                nA,
                np.mean(dataA),
                np.var(dataA, ddof=1),
                nB,
                np.mean(dataB),
                np.var(dataB, ddof=1),
                test_case["alternative"],
            )
            pvalues_ab[i] = p

        empirical_power = np.mean(pvalues_ab < alpha)
        print(
            f"帰無仮説を正しく棄却できた割合 (期待値 ~{power}): {empirical_power:.4f}"
        )
        print()

        result_dict = {
            "test_case_name": test_case["name"],
            "distribution": test_case["dist"],
            "alternative": test_case["alternative"],
            "sample_size_A": int(nA),
            "sample_size_B": int(nB),
            "type_i_error_rate": float(type_i_error_rate),
            "empirical_power": float(empirical_power),
        }
        results.append(result_dict)

    output_path = result_dir.joinpath("t.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
