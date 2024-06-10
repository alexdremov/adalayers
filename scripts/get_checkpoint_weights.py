import wandb
from adalayers.models import AdaLayersForSequenceClassification

import matplotlib
import matplotlib.pyplot as plt

wandb.init()
artifact = wandb.use_artifact('alexdremov/adalayers/final_imdb_adalayers_model_best:v7', type='model')
artifact_dir = artifact.download()

model = AdaLayersForSequenceClassification.from_pretrained(artifact_dir)
distr = model.distribution_normalized.view(-1).cpu().detach().numpy()

font = {
    'size': 12 * 1.333
}
matplotlib.rc('font', **font)

plt.figure(figsize=(10, 5))
plt.bar(list(i + 1 for i in range(len(distr))), distr, color='teal',)
plt.xlabel("Номер слоя")
plt.ylabel("Величина $p_i$")

plt.savefig('adalayers_distribution.svg', transparent=True)
plt.savefig('adalayers_distribution.png', transparent=True)
