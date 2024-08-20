import json

from adalayers.models import AdaLayersForSequenceClassification

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

from final_info import all_entities, api, STRINGS, LANG


def get_info(name, checkpoint):
    artifact = api.artifact(checkpoint, type="model")
    print(name)
    print(artifact.description)
    print(json.dumps(artifact.metadata, indent=4))
    artifact_dir = artifact.download()

    model = AdaLayersForSequenceClassification.from_pretrained(artifact_dir)
    distr = model.distribution_normalized.view(-1).cpu().detach().numpy()
    model.config.to_json_file(f"final_configs/{name}.json", use_diff=True)

    added_params = sum(
        np.prod(v.shape)
        for k, v in model.named_parameters()
        if not k.startswith("model.")
    )
    base_params = sum(
        np.prod(v.shape) for k, v in model.named_parameters() if k.startswith("model.")
    )
    print("Num added params:", added_params)
    print("Num base params:", base_params)
    params_portion = added_params / base_params * 100
    print(f"Portion params: {params_portion:.1f}%")

    font = {"size": 12 * 1.333}
    matplotlib.rc("font", **font)

    plt.figure(figsize=(10, 6))
    plt.bar(
        list(i + 1 for i in range(len(distr))),
        distr,
        color="teal",
    )
    plt.xlabel(STRINGS['layer_number'])
    plt.ylabel("\\$\\tilde{p}_i\\$")

    loc = plticker.MultipleLocator(base=0.1)
    plt.gca().yaxis.set_major_locator(loc)

    plt.tight_layout()

    plt.savefig(f"distrs/{name}_distr{LANG}.svg", transparent=True)
    plt.savefig(f"distrs/{name}_distr{LANG}.png", transparent=True)

    metrics = {"fone": artifact.metadata["f1"], "acc": artifact.metadata["acc"]}
    metrics = {k: v if v > 1 else v * 100 for k, v in metrics.items()}
    name = name.replace("_", "").replace("-", "")
    return (
        [
            "\\newcommand{\\"
            + name
            + metric.capitalize()
            + "}{"
            + f"{value:.1f}\\%"
            + "}"
            for metric, value in metrics.items()
        ]
        + [
            "\\newcommand{\\"
            + name
            + "ParamsPercent}{"
            + f"{params_portion:.1f}\\%"
            + "}"
        ]
        + ["\\newcommand{\\" + name + "Params}{" + f"{added_params / 1e6:.2f}M" + "}"]
    )


latex_info = []

for entity in all_entities:
    latex_info += get_info(name=entity["name"], checkpoint=entity["model_artifact"])
print("\n".join(latex_info))
