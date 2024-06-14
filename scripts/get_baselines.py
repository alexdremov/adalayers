import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from final_info import (
    all_entities_mapping,
    api
)

font = {"size": 12 * 1.333}
matplotlib.rc("font", **font)


def get_baseline_info(baseline, num_layers, metric):
    layer_pick = int(baseline.name.split()[1])
    metric = baseline.summary[f"test_best_{metric}"]
    if metric < 1:
        metric *= 100
    return (num_layers + layer_pick + 1, metric)


def get_max_layers(baseline):
    history = baseline.history()
    all_layers = [
        int(k.replace("distribution/layer_", ""))
        for k in history.columns
        if k.startswith("distribution")
    ]
    return max(all_layers) + 1


def plot_baselines(name, baselines, metric, ours_score):
    baseline_data = [api.run(baseline) for baseline in baselines]
    baseline_layers = [get_max_layers(baseline) for baseline in baseline_data]
    num_layers = baseline_layers[0]
    assert all(num_layers == i for i in baseline_layers)

    baseline_data = [
        get_baseline_info(baseline=baseline, num_layers=num_layers, metric=metric)
        for baseline in baseline_data
    ]

    baseline_layers = np.array([layer for layer, _ in baseline_data])
    baseline_scores = [score for _, score in baseline_data]
    plt.figure(figsize=(10, 6))
    plt.plot(
        baseline_layers,
        baseline_scores,
        label="базовые решения",
        marker="x",
        linestyle="dotted",
        markersize=12,
        linewidth=4,
        color="teal",
        mew=3,
    )
    for i, value in enumerate(baseline_scores):
        plt.annotate(
            f"{value:.1f}", xy=(baseline_layers[i] - 0.17, baseline_scores[i] + 0.6)
        )

    plt.hlines(
        min(ours_score + 2.5, 100),
        xmin=baseline_layers.min(),
        xmax=baseline_layers.max(),
        color=(0, 0, 0, 0.01),
    )

    plt.hlines(
        y=ours_score,
        xmin=baseline_layers.min(),
        xmax=baseline_layers.max(),
        colors="crimson",
        label="предложенное решение",
        linewidth=4,
    )

    metric_name = {"acc": "accuracy, %", "f1": "$f_1$, %"}[metric]

    plt.legend(loc="upper right", fancybox=True, framealpha=1.0)
    plt.xlabel("Номер слоя")
    plt.ylabel(f"{metric_name}")

    plt.savefig(f"baselines/{name}.svg", transparent=True)
    plt.savefig(f"baselines/{name}.png", transparent=True)


plot_baselines(
    name="imdb",
    baselines=[
        "alexdremov/adalayers/80g5q554",
        "alexdremov/adalayers/l5747f71",
        "alexdremov/adalayers/vza6uc0r",
        "alexdremov/adalayers/ndcbtwq5",
        "alexdremov/adalayers/a8hghqex",
        "alexdremov/adalayers/3gi2jv25",
        "alexdremov/adalayers/zb2yzvwr",
        "alexdremov/adalayers/nhqox0ez",
    ],
    metric="acc",
    ours_score=all_entities_mapping['imdb']['metric'],
)

plot_baselines(
    name="conll",
    baselines=[
        "alexdremov/adalayers/pz0k9p56",
        "alexdremov/adalayers/ytn5db8i",
        "alexdremov/adalayers/j0c8azse",
        "alexdremov/adalayers/hzlhmo37",
        "alexdremov/adalayers/rbw7zqb9",
        "alexdremov/adalayers/s5e9h0pc",
        "alexdremov/adalayers/3tr0l2qs",
        "alexdremov/adalayers/w72x3lrh",
    ],
    metric="f1",
    ours_score=all_entities_mapping['conll']['metric'],
)
