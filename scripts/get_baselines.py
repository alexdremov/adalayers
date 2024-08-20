import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from final_info import all_entities_mapping, api, STRINGS, LANG

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
    baseline_data = [run for run in baseline_data if run.state != "running"]
    baseline_layers = [get_max_layers(baseline) for baseline in baseline_data]
    num_layers = baseline_layers[0]
    assert all(num_layers == i for i in baseline_layers)

    baseline_data = [
        get_baseline_info(baseline=baseline, num_layers=num_layers, metric=metric)
        for baseline in baseline_data
    ]
    baseline_data = sorted(baseline_data)

    baseline_layers = np.array([layer for layer, _ in baseline_data])
    baseline_scores = [score for _, score in baseline_data]

    def plot(metric_names, prefix=''):
        plt.figure(figsize=(10, 6))
        plt.plot(
            baseline_layers,
            baseline_scores,
            label=STRINGS["base_solutions"],
            marker="x",
            linestyle="dotted",
            markersize=12,
            linewidth=4,
            color="teal",
            mew=3,
        )
        for i, value in enumerate(baseline_scores):
            plt.annotate(
                f"{value:.1f}",
                xy=(baseline_layers[i], baseline_scores[i] + 0.6),
                ha="center",
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
            label=STRINGS['proposed_solution'],
            linewidth=4,
        )

        metric_name = metric_names[metric]

        plt.legend(loc="upper right", fancybox=True, framealpha=1.0)
        plt.xlabel(STRINGS['layer_number'], labelpad=12)
        plt.ylabel(f"{metric_name}", labelpad=12)
        plt.xticks(baseline_layers.tolist())
        plt.tight_layout()

        plt.savefig(f"baselines/{name}{prefix}{LANG}.svg", transparent=True)
        plt.savefig(f"baselines/{name}{prefix}{LANG}.png", transparent=True)
        plt.close()


    metric_names = {"acc": "accuracy, %", "f1": "\\$f_1\\$, %"}
    plot(metric_names)

    metric_names = {"acc": "accuracy, %", "f1": "$f_1$, %"}
    plot(metric_names, '_rendered')


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
    ours_score=all_entities_mapping["imdb"]["metric"],
)

plot_baselines(
    name="cola",
    baselines=[
        "alexdremov/adalayers/d1xs63xn",
        "alexdremov/adalayers/o1gl9jit",
        "alexdremov/adalayers/5ssyxsks",
        "alexdremov/adalayers/12lsg2ed",
        "alexdremov/adalayers/3z3p00m1",
        "alexdremov/adalayers/7jgo733w",
        "alexdremov/adalayers/d7v50xjf",
        "alexdremov/adalayers/zm3v60fz",
        "alexdremov/adalayers/clcehodm",
        "alexdremov/adalayers/x4p007ty",
        "alexdremov/adalayers/18a0yluz",
        "alexdremov/adalayers/ceb00j03",
    ],
    metric="acc",
    ours_score=all_entities_mapping["cola"]["metric"],
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
        "alexdremov/adalayers/n3ue81hz",
        "alexdremov/adalayers/sdq1hg1r",
    ],
    metric="f1",
    ours_score=all_entities_mapping["conll"]["metric"],
)
