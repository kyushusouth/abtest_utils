import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.hypothesis_test.t import welch_t_test, welch_t_test_cuped
from src.sample_size.t import welch_t_test_sample_size_cuped


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
    rho = 0.7

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
    ]

    results = []

    for test_case in test_cases:
        print(f"--- {test_case['name']} ---")

        cov = rho * math.sqrt(vA) * math.sqrt(test_case["vB"])

        # サンプルサイズの計算
        if test_case["alternative"] == "two-sided":
            nA, nB = welch_t_test_sample_size_cuped(
                mA,
                test_case["mB"],
                vA,
                test_case["vB"],
                rho,
                test_case["alternative"],
                alpha,
                power,
                alloc_ratio,
            )
        elif test_case["alternative"] in ("greater", "smaller"):
            nA, nB = welch_t_test_sample_size_cuped(
                mA,
                test_case["mB"],
                vA,
                test_case["vB"],
                rho,
                "one-sided",
                alpha,
                power,
                alloc_ratio,
            )
        else:
            raise ValueError("Not Implemented")

        print(f"必要サンプルサイズ: nA = {nA}, nB = {nB}")

        # 第一種の誤り発生確率がα程度であることを確認
        pvalues_aa_default = np.zeros(n_sim)
        pvalues_aa_cuped = np.zeros(n_sim)
        for i in tqdm(range(n_sim)):
            rng = np.random.default_rng(i)

            # サンプルサイズ計算に用いた相関係数から計算した共分散を指定する
            dataA = rng.multivariate_normal([mA, mA], [[vA, cov], [cov, vA]], nA)
            y_A, x_A = dataA[:, 0], dataA[:, 1]
            dataB = rng.multivariate_normal([mA, mA], [[vA, cov], [cov, vA]], nB)
            y_B, x_B = dataB[:, 0], dataB[:, 1]

            _, p_default = welch_t_test(
                nA,
                np.mean(y_A),
                np.var(y_A, ddof=1),
                nB,
                np.mean(y_B),
                np.var(y_B, ddof=1),
                test_case["alternative"],
            )
            pvalues_aa_default[i] = p_default

            y_all = np.concatenate([y_A, y_B])
            x_all = np.concatenate([x_A, x_B])
            sample_rho = np.corrcoef(y_all, x_all)[0, 1]
            _, p_cuped = welch_t_test_cuped(
                nA,
                np.mean(y_A),
                np.mean(x_A),
                np.var(y_A, ddof=1),
                nB,
                np.mean(y_B),
                np.mean(x_B),
                np.var(y_B, ddof=1),
                sample_rho,
                test_case["alternative"],
            )
            pvalues_aa_cuped[i] = p_cuped

        type_i_error_rate_default = np.mean(pvalues_aa_default < alpha)
        type_i_error_rate_cuped = np.mean(pvalues_aa_cuped < alpha)
        print(
            f"帰無仮説を誤って棄却した割合 (CUPEDなし): {type_i_error_rate_default:.4f}"
        )
        print(
            f"帰無仮説を誤って棄却した割合 (CUPEDあり, 期待値 ~{alpha}): {type_i_error_rate_cuped:.4f}"
        )

        # 検出力が(1-β)程度であることを確認
        pvalues_ab_default = np.zeros(n_sim)
        pvalues_ab_cuped = np.zeros(n_sim)
        for i in tqdm(range(n_sim)):
            rng = np.random.default_rng(i)

            # サンプルサイズ計算に用いた相関係数から計算した共分散を指定する
            dataA = rng.multivariate_normal([mA, mA], [[vA, cov], [cov, vA]], nA)
            y_A, x_A = dataA[:, 0], dataA[:, 1]
            dataB = rng.multivariate_normal(
                [test_case["mB"], mA], [[vA, cov], [cov, vA]], nB
            )
            y_B, x_B = dataB[:, 0], dataB[:, 1]

            _, p_default = welch_t_test(
                nA,
                np.mean(y_A),
                np.var(y_A, ddof=1),
                nB,
                np.mean(y_B),
                np.var(y_B, ddof=1),
                test_case["alternative"],
            )
            pvalues_ab_default[i] = p_default

            pooled_y = np.concatenate([y_A, y_B])
            pooled_x = np.concatenate([x_A, x_B])
            sample_rho = np.corrcoef(pooled_y, pooled_x)[0, 1]
            _, p_cuped = welch_t_test_cuped(
                nA,
                np.mean(y_A),
                np.mean(x_A),
                np.var(y_A, ddof=1),
                nB,
                np.mean(y_B),
                np.mean(x_B),
                np.var(y_B, ddof=1),
                sample_rho,
                test_case["alternative"],
            )
            pvalues_ab_cuped[i] = p_cuped

        empirical_power_default = np.mean(pvalues_ab_default < alpha)
        empirical_power_cuped = np.mean(pvalues_ab_cuped < alpha)
        print(
            f"帰無仮説を正しく棄却できた割合 (CUPEDなし): {empirical_power_default:.4f}"
        )
        print(
            f"帰無仮説を正しく棄却できた割合 (CUPEDあり, 期待値 ~{power}): {empirical_power_cuped:.4f}"
        )
        print()

        result_dict = {
            "test_case_name": test_case["name"],
            "distribution": test_case["dist"],
            "alternative": test_case["alternative"],
            "type_i_error_rate_default": float(type_i_error_rate_default),
            "type_i_error_rate_cuped": float(type_i_error_rate_cuped),
            "empirical_power_default": float(empirical_power_default),
            "empirical_power_cuped": float(empirical_power_cuped),
        }
        results.append(result_dict)

    output_path = result_dir.joinpath("cuped.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
