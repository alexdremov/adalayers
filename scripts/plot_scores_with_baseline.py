import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {
    'size': 12 * 1.333
}
matplotlib.rc('font', **font)

baseline_layers = np.array([
    19, 20, 21, 22, 23, 24
])
baseline_scores = [
    91.30, 90.00, 89.71, 90.01, 85.09, 85.09
]
plt.figure(figsize=(10, 5))
plt.plot(
    baseline_layers,
    baseline_scores,
    label="базовые решения",
    marker='o',
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
