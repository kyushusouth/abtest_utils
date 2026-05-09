from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

project_dir = Path(__file__).parent.parent.parent
save_path = project_dir.joinpath(
    "data", "hypothesis_test_simulation", "binom_normal_comparison.png"
)
save_path.parent.mkdir(parents=True, exist_ok=True)

p = 0.5
n_list = [10, 50, 100, 1000]

plt.figure(figsize=(12, 6), dpi=300)

for i, n in enumerate(n_list):
    x = np.arange(stats.binom.ppf(0.0001, n, p), stats.binom.ppf(0.9999, n, p))

    pmf_binom = stats.binom.pmf(x, n, p)

    mu = n * p
    sigma = np.sqrt(n * p * (1 - p))
    x_norm = np.linspace(min(x), max(x), 10000)
    pdf_norm = stats.norm.pdf(x_norm, mu, sigma)

    plt.subplot(1, 4, i + 1)
    plt.bar(x, pmf_binom, alpha=0.5, label=f"Binomial (n={n})")
    plt.plot(x_norm, pdf_norm, "k-", lw=2, label="Normal Approx")

    plt.title(f"n = {n}")
    plt.xlabel("Successes")
    plt.ylabel("Probability")
    plt.legend()

plt.tight_layout()
plt.savefig(save_path)
