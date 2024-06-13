import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {
    'size': 12 * 1.333
}
matplotlib.rc('font', **font)

baseline_data = [
    (-1, 0.85088),
    (-2, 0.83324),
    (-3, 0.90072),
    (-4, 0.89708),
    (-5, 0.89768),
    (-6, 0.91300),
    (-7, 0.89572),
    (-8, 0.89484),
]

baseline_layers = np.array([25 + layer for layer, _ in baseline_data])
baseline_scores = [score * 100 for _, score in baseline_data]
plt.figure(figsize=(10, 5))
plt.plot(
    baseline_layers,
    baseline_scores,
    label="базовые решения",
    marker='x',
    linestyle='dotted',
    markersize=12,
    linewidth=4,
    color='teal',
)
for i, value in enumerate(baseline_scores):
    plt.annotate(f"{value:.1f}", xy=(baseline_layers[i] - 0.17, baseline_scores[i] + 0.6))

plt.hlines(
    y=96.1,
    xmin=baseline_layers.min(),
    xmax=baseline_layers.max(),
    colors='crimson',
    label="предложенное решение",
    linewidth=4,
)

plt.legend(loc='lower left')
plt.xlabel("Номер слоя")
plt.ylabel("accuracy, %")

plt.savefig('baselines_compare.svg', transparent=True)
plt.savefig('baselines_compare.png', transparent=True)
