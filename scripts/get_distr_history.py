import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from final_info import (
    all_entities,
    api
)

sns.set_style('dark')


def get_info(name, run):
    run = api.run(run)
    history = run.history()
    distribution_names = [k for k in history.columns if k.startswith("distribution")]
    distributions = history[distribution_names]
    distributions.columns = distributions.columns.map(lambda x: x.replace('distribution/layer_', ''))
    distributions = distributions.dropna(axis=0)

    font = {
        'size': 12 * 1.333
    }
    matplotlib.rc('font', **font)

    plt.figure(figsize=(10, 6))

    steps = distributions.index
    for i, column in enumerate(distributions.columns):
        sns.lineplot(x=steps, y=distributions[column], linestyle=['solid', 'dotted', 'dashed'][i % 3])

    plt.xlabel("Шаг")
    plt.ylabel("Величина $\\tilde{{p}}_i$")

    plt.savefig(f'distrs_history/{name}_history.svg', transparent=True)
    plt.savefig(f'distrs_history/{name}_history.png', transparent=True)


for entity in all_entities:
    get_info(name=entity['name'], run=entity['run_name'])
