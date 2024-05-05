import wandb
from adalayers.models import AdaLayersForSequenceClassification

import matplotlib.pyplot as plt

wandb.init()
artifact = wandb.use_artifact('alexdremov/adalayers/final_glue_cola_adalayers_model_best:v0', type='model')
artifact_dir = artifact.download()

model = AdaLayersForSequenceClassification.from_pretrained(artifact_dir)
distr = model.distribution_normalized.view(-1).cpu().detach().numpy()

plt.bar(list(range(len(distr))), distr)
plt.savefig('adalayers_distribution.png')
