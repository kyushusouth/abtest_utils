from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t

project_dir = Path(__file__).parent.parent.parent
save_path = project_dir.joinpath(
    "data", "hypothesis_test_simulation", "t_norm_comparion.png"
)
save_path.parent.mkdir(parents=True, exist_ok=True)

# 1. 設定
x = np.linspace(-5, 5, 1000)
dfs = [5, 10, 30, 50, 100, 1000, 10000, 100000]

plt.figure(figsize=(10, 8), dpi=300)

# 2. 各自由度のt分布をプロット
for df in dfs:
    plt.plot(x, t.pdf(x, df), label=f"t-dist (df={df})", alpha=0.7)

# 3. 標準正規分布をプロット（比較対象）
plt.plot(x, norm.pdf(x), "k--", lw=2, label="Normal Dist (Standard)")

# 4. グラフの装飾
plt.title("Comparison of t-distribution and Normal distribution", fontsize=14)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(save_path)
