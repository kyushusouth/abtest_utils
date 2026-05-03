import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.hypothesis_test.proportion import proportion_diff_test
from src.sample_size.proportion import proportion_diff_test_sample_size


def main():
    project_dir = Path(__file__).parent.parent.parent
    result_dir = project_dir.joinpath("data", "hypothesis_test_simulation")
    result_dir.mkdir(parents=True, exist_ok=True)

    # パラメータの設定
    pA = 0.1
    alpha = 0.05
    power = 0.8
    alloc_ratio = 1.0
    n_sim = 10000

    # テストケースの定義
    test_cases = [
        {
            "name": "両側検定",
            "alternative": "two-sided",
            "pB": pA * 1.02,
        },
        {
            "name": "片側検定（A<B）",
            "alternative": "greater",
            "pB": pA * 1.02,
        },
        {
            "name": "片側検定（A>B）",
            "alternative": "smaller",
            "pB": pA * 0.98,
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"--- {test_case['name']} ---")

        # サンプルサイズの計算
        if test_case["alternative"] == "two-sided":
            nA, nB = proportion_diff_test_sample_size(
                pA, test_case["pB"], test_case["alternative"], alpha, power, alloc_ratio
            )
        elif test_case["alternative"] in ("greater", "smaller"):
            nA, nB = proportion_diff_test_sample_size(
                pA, test_case["pB"], "one-sided", alpha, power, alloc_ratio
            )
        else:
            raise ValueError("Not Implemented")

        print(f"必要サンプルサイズ: nA = {nA}, nB = {nB}")

        # 第一種の誤り発生確率がα程度であることを確認
        pvalues_aa = np.zeros(n_sim)
        for i in tqdm(range(n_sim)):
            rng = np.random.default_rng(i)

            # 同じ分布からデータを生成
            occurrenceA = rng.binomial(nA, pA)
            occurrenceB = rng.binomial(nB, pA)

            _, p = proportion_diff_test(
                nA, occurrenceA, nB, occurrenceB, test_case["alternative"]
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
            occurrenceA = rng.binomial(nA, pA)
            occurrenceB = rng.binomial(nB, test_case["pB"])

            _, p = proportion_diff_test(
                nA, occurrenceA, nB, occurrenceB, test_case["alternative"]
            )
            pvalues_ab[i] = p

        empirical_power = np.mean(pvalues_ab < alpha)
        print(
            f"帰無仮説を正しく棄却できた割合 (期待値 ~{power}): {empirical_power:.4f}"
        )
        print()

        result_dict = {
            "test_case_name": test_case["name"],
            "alternative": test_case["alternative"],
            "sample_size_A": int(nA),
            "sample_size_B": int(nB),
            "type_i_error_rate": float(type_i_error_rate),
            "empirical_power": float(empirical_power),
        }
        results.append(result_dict)

    output_path = result_dir.joinpath("proportion.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
