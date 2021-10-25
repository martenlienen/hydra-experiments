from pathlib import Path

import matplotlib.pyplot as pp

from utils import load_results

df = load_results("results")
df = df.set_index("maxfun").sort_index()

fig = pp.figure(figsize=(5, 3.5))
ax = fig.add_subplot(111)
ax.set_title("Minimizing the Rosenbrock function")
df.groupby("method.name")["f(x)"].plot(
    legend=True,
    logy=True,
    xlabel="Max Iterations",
    ylabel="$f(x_{\mathrm{opt}})$",
    ax=ax,
)

plots_dir = Path("figures")
plots_dir.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(plots_dir / f"results.png", dpi=90)
pp.show()
