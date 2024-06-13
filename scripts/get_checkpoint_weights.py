import wandb
from adalayers.models import AdaLayersForSequenceClassification

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

wandb.init()

def get_info(name, checkpoint):
    artifact = wandb.use_artifact(checkpoint, type='model')
    print(name)
    print(artifact.description)
    print(artifact.metadata)
    
    artifact_dir = artifact.download()

    model = AdaLayersForSequenceClassification.from_pretrained(artifact_dir)
    distr = model.distribution_normalized.view(-1).cpu().detach().numpy()

    added_params = sum(np.prod(v.shape) for k, v in model.named_parameters() if not k.startswith('model.'))
    base_params = sum(np.prod(v.shape) for k, v in model.named_parameters() if k.startswith('model.'))
    print("Num added params:", added_params)
    print("Num base params:", base_params)
    print("Portion params:", added_params / base_params)

    font = {
        'size': 12 * 1.333
    }
    matplotlib.rc('font', **font)

    plt.figure(figsize=(10, 5))
    plt.bar(list(i + 1 for i in range(len(distr))), distr, color='teal',)
    plt.xlabel("Номер слоя")
    plt.ylabel("Величина $p_i$")

    plt.savefig(f'{name}_distr.svg', transparent=True)
    plt.savefig(f'{name}_distr.png', transparent=True)
