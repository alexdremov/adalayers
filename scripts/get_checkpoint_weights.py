
import json

from adalayers.models import AdaLayersForSequenceClassification

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from final_info import (
    all_entities,
    api
)

def get_info(name, checkpoint):
    artifact = api.artifact(checkpoint, type='model')
    print(name)
    print(artifact.description)
    print(json.dumps(artifact.metadata, indent=4))
    artifact_dir = artifact.download()

    model = AdaLayersForSequenceClassification.from_pretrained(artifact_dir)
    distr = model.distribution_normalized.view(-1).cpu().detach().numpy()

    added_params = sum(np.prod(v.shape) for k, v in model.named_parameters() if not k.startswith('model.'))
    base_params = sum(np.prod(v.shape) for k, v in model.named_parameters() if k.startswith('model.'))
    print("Num added params:", added_params)
    print("Num base params:", base_params)
    params_portion = added_params / base_params * 100
    print(f"Portion params: {params_portion:.1f}%")

    font = {
        'size': 12 * 1.333
    }
    matplotlib.rc('font', **font)

    plt.figure(figsize=(10, 6))
    plt.bar(list(i + 1 for i in range(len(distr))), distr, color='teal',)
    plt.xlabel("Номер слоя")
    plt.ylabel(f"Величина $\\tilde{{p}}_i$")

    plt.savefig(f'distrs/{name}_distr.svg', transparent=True)
    plt.savefig(f'distrs/{name}_distr.png', transparent=True)

    metrics = {
        'f1': artifact.metadata['f1'],
        'acc': artifact.metadata['acc']
    }
    metrics = {
        k: v if v > 1 else v * 100 for k, v in metrics.items()
    }
    return [
        '\\newcommand{\\' + name + metric.capitalize() + '}{' + f'{value:.1f}\\%' + '}' for metric, value in metrics.items()
    ] + [
        '\\newcommand{\\' + name + 'ParamsPercent}{' + f'{params_portion:.1f}\\%' + '}'
    ] + [
        '\\newcommand{\\' + name + 'Params}{' + f'{added_params / 1e6:.2f}M' + '}'
    ]

latex_info = []

for entity in all_entities:
    latex_info += get_info(
        name=entity['name'],
        checkpoint=entity['model_artifact']
    )
print('\n'.join(latex_info))
